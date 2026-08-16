"""A small CNN classifier over log-mel spectrograms.

Baseline architecture: a few conv/pool blocks over the (mel, time) "image",
global average pooling, then a linear head. Cheap to train, reasonable
starting point before reaching for a CRNN or pretrained backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DanceCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()

        def conv_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
