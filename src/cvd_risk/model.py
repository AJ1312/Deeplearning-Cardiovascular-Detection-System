"""PyTorch model definitions for CVD risk prediction."""

import torch
import torch.nn as nn


class HeartDiseaseNet(nn.Module):
    """Deep neural network for binary heart disease prediction."""

    def __init__(self, input_size: int = 13) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
