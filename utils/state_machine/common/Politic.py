import numpy as np
import torch
from typing import Dict


class Politic:
    def __init__(self, environmentName: str, name: str) -> None:
        self.environmentName = environmentName
        self.name = name
        self.transitionNeed = False
        self.cuda = False

    def act(self, observation: np.ndarray) -> int:
        raise NotImplementedError

    def update(
        self,
        weights: Dict,
    ) -> None:
        raise NotImplementedError

    def actionDistribution(
        self,
        observation: np.ndarray,
    ) -> torch.Tensor:
        raise NotImplementedError
