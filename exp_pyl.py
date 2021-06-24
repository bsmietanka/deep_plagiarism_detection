import numpy as np
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from encoders.classifier import Classifier
from datasets.functions_dataset import FunctionsDataset
from typing import Optional
from typing_extensions import Literal

import torch
from torch import Tensor
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from torch import nn
from torch_geometric.data import Batch, DataLoader
from pytorch_metric_learning import distances, losses, miners, samplers, utils, regularizers
from numba import njit
from sklearn.metrics import accuracy_score, f1_score

from utils.train_utils import get_model


@njit(cache=True)
def find_first(vec, item):
    """return the index of the first occurence of item in vec"""
    for i, v in enumerate(vec):
        if item == v:
            return i
    return -1


class PlagiarismFunctions(LightningDataModule):
    def __init__(self, root_dir: str, format: Literal["graph", "tokens", "letters"], cache: Optional[str] = None):
        super().__init__()
        self.root_dir = root_dir
        self.format = format
        self.cache = cache
        self.batch_size = 128


    @property
    def num_features(self) -> int:
        return self.train_dataset.num_tokens


    def setup(self, stage: Optional[str] = None):
        self.train_dataset = FunctionsDataset(self.root_dir, "train.txt", "pairs", self.format, cache=self.cache, multiplier=10)
        self.val_dataset = FunctionsDataset(self.root_dir, "val.txt", "singles", self.format, 50, cache=self.cache)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=12)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=12)


    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=12)



class PlagiarismModel(LightningModule):
    def __init__(self, model: Classifier):
        super().__init__()

        self.model = model

        self.distance = distances.CosineSimilarity()
        # TODO?: Try other losses, eg. ContrastiveLoss
        self.emb_loss = losses.TripletMarginLoss(margin=0.2, distance=self.distance, embedding_regularizer=regularizers.LpRegularizer())
        self.cls_loss = nn.BCELoss()

        self.miner = miners.TripletMarginMiner(type_of_triplets="all", distance=self.distance)

        self.emb_acc = utils.accuracy_calculator.AccuracyCalculator(include=("mean_average_precision_at_r",))
        self.val_acc = Accuracy()
        self.train_acc = Accuracy()

        self.cache = {}


    def forward(self, batch: Batch) -> Tensor:
        return self.model(batch)


    def training_step(self, batch: tuple, batch_idx: int):
        embs = self(batch[0])
        aug_embs = self(batch[1])
        
        embeddings = torch.cat([embs, aug_embs], dim=0)
        labels = torch.cat([batch[2], batch[3]], dim=0)

        indices_tuple = self.miner(embeddings, labels)

        emb_loss = self.emb_loss(embeddings, labels, indices_tuple)

        a = embeddings[indices_tuple[0]]
        p = embeddings[indices_tuple[1]]
        n = embeddings[indices_tuple[2]]
        pred_p = self.model.classify_embs(a, p)
        pred_n = self.model.classify_embs(a, n)
        preds = torch.cat([pred_p, pred_n], dim=0)
        y_true = torch.cat([torch.ones_like(pred_p), torch.zeros_like(pred_n)], dim=0)
        cls_loss = self.cls_loss(preds, y_true)

        loss = cls_loss + emb_loss

        self.train_acc(preds, y_true.long())
        self.log('acc/train', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        embs = self(batch[0])
        return {"embeddings": embs, "labels": batch[1]}


    def validation_epoch_end(self, outputs) -> None:
        embeddings = np.concatenate([o["embeddings"].cpu().numpy() for o in outputs], axis=0)
        labels = np.squeeze(np.concatenate([o["labels"].cpu().numpy() for o in outputs], axis=0))

        labels_vals = np.unique(labels)
        idxs = []
        for v in labels_vals:
            idxs.append(find_first(labels, v))
        mask = np.ones(len(labels)).astype(bool)
        mask[idxs] = False

        accuracies = self.emb_acc.get_accuracy(embeddings[mask],
                                               embeddings[~mask],
                                               labels[mask],
                                               labels[~mask],
                                               False)

        if "triplets" not in self.cache:
            self.cache["triplets"] = torch.tensor(samplers.FixedSetOfTriplets(labels, 100000).fixed_set_of_triplets)

        triplets = self.cache["triplets"]
        a = triplets[:, 0]
        p = triplets[:, 1]
        n = triplets[:, 2]
        accuracies["emb_loss"] = self.emb_loss(torch.tensor(embeddings), torch.tensor(labels), (a, p, a, n)).item()

        embeddings = torch.tensor(embeddings)
        labels = torch.tensor(labels)

        a = embeddings[a]
        p = embeddings[p]
        n = embeddings[n]

        cls_dataset = TensorDataset(a, p, n)
        cls_dataloader = DataLoader(cls_dataset, 512, num_workers=10)
        prob = []
        y_true = []
        for ba, bp, bn in cls_dataloader:
            ba = ba.to(self.device)
            prob_p = self.model.classify_embs(ba, bp.to(self.device)).cpu().numpy()
            prob_n = self.model.classify_embs(ba, bn.to(self.device)).cpu().numpy()
            y_true.extend([np.ones_like(prob_p), np.zeros_like(prob_n)])
            prob.extend([prob_p, prob_n])
        y_true = np.concatenate(y_true)
        prob = np.concatenate(prob)
        pred = prob > 0.5
        accuracies["cls_loss"] = self.cls_loss(torch.tensor(prob), torch.tensor(y_true)).item()
        accuracies["accuracy"] = accuracy_score(pred, y_true)
        accuracies["f1_score"] = f1_score(pred, y_true)

        self.log("loss/val", accuracies["cls_loss"] + accuracies["emb_loss"], prog_bar=True)
        self.log("acc/val", accuracies["accuracy"], prog_bar=True)
        self.log("f1/val", accuracies["f1_score"], prog_bar=True)
        self.log("MAP@R/val", accuracies["mean_average_precision_at_r"], prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'loss/val',
            }
        }


def main():
    seed_everything(42)
    dm = PlagiarismFunctions(root_dir="data/graph_functions/", format="graph", cache="data/cache")
    dm.setup()
    model_params = {
        "input_dim": 1,
        "node_embeddings": 4,
        "hidden_dim": 512,
        "num_layers": 5
    }
    m = get_model("gnn", dm.train_dataset, 1, "cpu", **model_params)
    model = PlagiarismModel(m)
    checkpoint_callback = ModelCheckpoint(monitor='loss/val', save_top_k=1)
    early_stopping = EarlyStopping("loss/val", patience=20)
    trainer = Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback, early_stopping], stochastic_weight_avg=True)

    # Uncomment to train on multiple GPUs:
    # trainer = Trainer(gpus=2, accelerator='ddp', max_epochs=20,
    #                   callbacks=[checkpoint_callback])

    trainer.fit(model, train_dataloader=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    model = PlagiarismModel.load_from_checkpoint(checkpoint_callback.best_model_path, model=m)
    torch.save(model.model.state_dict(), "exp_pyl.pt")


if __name__ == "__main__":
    main()