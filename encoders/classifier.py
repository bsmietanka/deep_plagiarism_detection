from argparse import ArgumentError
import torch
from torch import nn


class Classifier(nn.Module):

    def __init__(self, encoder: nn.Module, num_classes: int = 1):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_classes)


    def forward(self, x):
        return self.encoder(x)


    def classify_embs(self, *args):
        if len(args) == 2:
            x, y = args
            diff = torch.abs(x - y)

            if self.num_classes == 1:
                return torch.sigmoid(self.classifier(diff))
            else:
                return torch.log_softmax(self.classifier(diff), -1)
        elif len(args) == 1:
            x = args[0]
            if self.num_classes == 1:
                return torch.sigmoid(self.classifier(x))
            else:
                return torch.log_softmax(self.classifier(x), -1)
        else:
            raise ArgumentError("Too many arguments passed to function")


    def classify(self, *args):
        embs = []
        for x in args:
            embs.append(self(x))

        return self.classify_embs(*embs)
