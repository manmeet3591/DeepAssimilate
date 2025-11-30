# deepassimilate/assimilation/observation_ops.py
from abc import ABC, abstractmethod
import torch


class ObservationOperator(ABC):
    """
    Maps a full state x to observation space H(x).

    Implement forward() for your observation mapping (e.g., masking grid points,
    projecting to a subset, etc.).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class IdentityObservationOperator(ObservationOperator):
    """
    H(x) = x.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
