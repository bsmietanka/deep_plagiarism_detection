import torch

class Accuracy:

    def __init__(self, threshold: float = 0.5):
        self.reset()
        self.threshold = threshold

    def __call__(self, probs: torch.Tensor, target: torch.Tensor):
        target = target.int()
        preds = probs >= self.threshold
        self.correct += (preds == target).sum().item()
        self.all += target.shape[0]
        self.acc = self.correct / self.all

    def reset(self):
        self.all = 0
        self.correct = 0
        self.acc = None

    def __str__(self):
        return str(self.acc)

    def get(self) -> float:
        return self.acc

    @property
    def name(self) -> str:
        return "accuracy"
