
from typing import Optional, Union
from pprint import pprint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from utils.train_utils import create_triplet_dataset, get_model


import torch
from typing_extensions import Literal
from pytorch_lightning import LightningDataModule, LightningModule
from torch_geometric.data import Batch, Data
from torch import Tensor
from torch import nn
from pytorch_metric_learning import samplers
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm

from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
from encoders.classifier import Classifier
from encoders.graph_encoder import GraphEncoder

from pytorch_lightning import seed_everything
seed_everything(42)


class PlagiarismFunctions(LightningDataModule):
    def __init__(self, root_dir: str, format: Literal["graph", "tokens", "letters"], cache: Optional[str] = None):
        super().__init__()
        self.root_dir = root_dir
        self.format = format
        self.cache = cache
        self.batch_size = 64
        train_dataset = FunctionsDataset(self.root_dir, "train.txt", "singles", self.format)
        self._num_features = train_dataset.num_tokens
        val_dataset = FunctionsDataset(self.root_dir, "val.txt", "singles", self.format)


        train_triplets = torch.tensor(samplers.FixedSetOfTriplets(train_dataset.labels, 10000 * 50).fixed_set_of_triplets)
        self.train_dataset = create_triplet_dataset(train_dataset, train_triplets)
        val_triplets = torch.tensor(samplers.FixedSetOfTriplets(val_dataset.labels, 10**4).fixed_set_of_triplets)
        self.val_dataset = create_triplet_dataset(val_dataset, val_triplets)

    @property
    def num_features(self) -> int:
        return self._num_features

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=20_000), pin_memory=True, num_workers=12, collate_fn=Collater())


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=12, collate_fn=Collater())


    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=12, collate_fn=Collater())



class PlagiarismModel(LightningModule):
    def __init__(self, model: Classifier):
        super().__init__()
        self.model = model
        self.cls_loss = nn.BCELoss()


    def forward(self, x1, x2) -> Tensor:
        return self.model.classify(x1, x2)


    def training_step(self, batch: tuple, batch_idx: int):
        pos = self(batch[0], batch[1])
        neg = self(batch[0], batch[2])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
        preds = torch.cat([pos, neg], dim=0)
        loss = self.cls_loss(preds, labels)

        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        pos = self(batch[0], batch[1])
        neg = self(batch[0], batch[2])
        probs = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)

        correct = torch.sum(pos > 0.5).item()
        correct += torch.sum(neg < 0.5).item()
        acc = correct / (2 * pos.shape[0])
        loss = self.cls_loss(probs, labels).item()
        self.log("loss/val", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("acc/val", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'loss/val',
            }
        }


def main():
    dm = PlagiarismFunctions(root_dir="data/graph_functions/", format="graph", cache="data/cache")
    dm.setup()
    m = Classifier(GraphEncoder(1, 1024, node_labels=dm.num_features, num_layers=5, node_embeddings=5, train_eps=False))
    model = PlagiarismModel(m)
    checkpoint_callback = ModelCheckpoint(monitor='loss/val', save_top_k=1)
    early_stopping = EarlyStopping("loss/val", patience=20)
    trainer = Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback, early_stopping])

    # Uncomment to train on multiple GPUs:
    # trainer = Trainer(gpus=2, accelerator='ddp', max_epochs=20,
    #                   callbacks=[checkpoint_callback, early_stopping])

    trainer.fit(model, train_dataloader=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    model = PlagiarismModel.load_from_checkpoint(checkpoint_callback.best_model_path, model=m)
    torch.save(model.model.state_dict(), "exp_pyl.pt")


if __name__ == '__main__':
    main()
