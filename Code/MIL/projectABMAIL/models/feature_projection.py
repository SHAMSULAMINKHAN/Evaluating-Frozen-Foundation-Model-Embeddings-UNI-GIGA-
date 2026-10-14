import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    Project foundation model features into a lower-dimensional
    embedding space for MIL.

    Input:
        [N, 1536]

    Output:
        [N, 512]
    """

    def __init__(
        self,
        input_dim=1536,
        output_dim=512,
        dropout=0.25,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x:
            [N,1536]

        returns:
            [N,512]
        """
        return self.projection(x)