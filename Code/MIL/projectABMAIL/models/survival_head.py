import torch
import torch.nn as nn


class SurvivalHead(nn.Module):
    """
    Cox proportional hazards prediction head.

    Input:
        [1, 512]

    Output:
        [1, 1] risk score
    """

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        dropout=0.25,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Args
        ----
        x : [1,512]

        Returns
        -------
        risk : [1,1]
        """
        return self.head(x)