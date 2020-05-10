from abc import ABC, abstractmethod
import torch

class RunningMetric(ABC):

    def __init__(self, objective):
        self.multilabel = True if objective == 'multilabel' else False

    @abstractmethod
    def __call__(self, probs: torch.Tensor, target: torch.Tensor):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def get(self):
        ...

    @abstractmethod
    def name(self):
        ...

    def _probs_to_preds(self, probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        if self.multilabel:
            return probs > threshold
        return probs.max(1)[1]

    @abstractmethod
    def __str__(self):
        ...
