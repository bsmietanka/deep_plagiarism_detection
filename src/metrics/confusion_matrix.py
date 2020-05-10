import torch
from sklearn.metrics import confusion_matrix
import numpy as np

class ConfusionMatrix:

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.reset()

    def __call__(self, probs: torch.Tensor, target: torch.Tensor):
        gts = target.cpu().numpy().tolist()
        preds = (probs >= self.threshold).cpu().numpy().tolist()

        for gt, pr in zip(gts, preds):
            self.res['targets'].append(gt)
            self.res['predictions'].append(pr)

    def _get_confusion_matrix(self):
        return confusion_matrix(self.res['targets'], self.res['predictions'])

    def reset(self):
        self.res = {"targets": [], "predictions": []}

    def get(self):
        return self._get_confusion_matrix()

    # TODO: pretty print
    def __str__(self):
        return "\n" + str(self._get_confusion_matrix())

    @property
    def name(self) -> str:
        return "confussion matrix"
