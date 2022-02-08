from typing import Literal, Union
import torch
from pytorch_lightning import LightningModule
from pytorch_metric_learning import distances, losses, miners, regularizers
from torch import nn, optim

from encoders.classifier import Classifier
from utils.train_utils import get_model


class PlagiarismModel(LightningModule):
    def __init__(self, classify: bool, loss_fun: Literal["triplet", "contrastive"] = "triplet",
                 distance: Literal["cosine", "euclidean"] = "euclidean", **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.classify = classify
        self.model: Union[nn.Module, Classifier] = get_model(**model_kwargs)
        if self.classify:
            self.model = Classifier(self.model)
            self.loss_fun = nn.BCELoss()

        else:
            if distance == "cosine":
                self.distance = distances.CosineSimilarity()
            else:
                self.distance = distances.LpDistance()
            if loss_fun == "triplet":
                self.loss_fun = losses.TripletMarginLoss(margin=0.05, distance=self.distance, embedding_regularizer=regularizers.LpRegularizer())
            else:
                self.loss_fun = losses.ContrastiveLoss(distance=self.distance)
            self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all", distance=self.distance)


    def forward(self, *args) -> torch.Tensor:
        if self.classify:
            return self.model.classify(*args)
        return self.model(*args)


    def training_step(self, batch: tuple, batch_idx: int):
        if self.classify:
            pos = self(batch[0], batch[1])
            neg = self(batch[0], batch[2])
            labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
            preds = torch.cat([pos, neg], dim=0)
            loss = self.loss_fun(preds, labels)

            self.log("loss/train", loss, prog_bar=True, on_epoch=True, on_step=False)
            return loss
        else:
            embs = self(batch[0])
            aug_embs = self(batch[1])

            embeddings = torch.cat([embs, aug_embs], dim=0)
            labels = torch.cat([batch[2], batch[3]], dim=0)

            indices_tuple = self.miner(embeddings, labels)
            loss = self.loss_fun(embeddings, labels, indices_tuple)

            # anchor_embs = self(batch[0])
            # pos_embs    = self(batch[1])
            # neg_embs    = self(batch[2])

            # embeddings = torch.cat([anchor_embs, pos_embs, neg_embs], dim=0)
            # anchor_ind = torch.arange(0, len(anchor_embs)).long()
            # pos_ind    = anchor_ind + len(anchor_embs)
            # neg_ind    = pos_ind    + len(anchor_embs)
            # labels = torch.cat(batch[3:], dim=0).long()

            # indices_tuple = (anchor_ind, pos_ind, anchor_ind, neg_ind)

            # loss = self.loss_fun(embeddings, labels, indices_tuple)

            self.log("loss/train", loss, prog_bar=True, on_epoch=True, on_step=False)
            return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        if self.classify:
            pos = self(batch[0], batch[1])
            neg = self(batch[0], batch[2])
            probs = torch.cat([pos, neg], dim=0)
            labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)

            loss = self.loss_fun(probs, labels)
            self.log("loss/val", loss, prog_bar=True, on_epoch=True, on_step=False)
            return loss
        else:
            embs_a = self(batch[0])
            embs_p = self(batch[1])
            embs_n = self(batch[2])
            return {"embs_a": embs_a, "embs_p": embs_p, "embs_n": embs_n}

    def validation_epoch_end(self, outputs) -> None:
        if self.classify:
            # super().validation_epoch_end(outputs)
            pass
        else:
            embs_a = torch.cat([o["embs_a"] for o in outputs], dim=0)
            embs_p = torch.cat([o["embs_p"] for o in outputs], dim=0)
            embs_n = torch.cat([o["embs_n"] for o in outputs], dim=0)

            embeddings = torch.cat([embs_a, embs_p, embs_n], dim=0)

            a = torch.arange(0, embs_a.shape[0])
            p = a + embs_a.shape[0]
            n = p + embs_a.shape[0]
            fake_labels = torch.cat([a, a, n], dim=0)
            loss = self.loss_fun(embeddings, fake_labels, (a, p, a, n)).item()

            self.log("loss/val", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            factor=0.3, patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'loss/val',
            }
        }

    def similarity(self, x1, x2):
        if self.classify:
            return self.model.classify(x1, x2)
        if self.distance.is_inverted:
            return self.distance.pairwise_distance(self.model(x1), self.model(x2))
        return 1 - self.distance.pairwise_distance(self.model(x1), self.model(x2))
