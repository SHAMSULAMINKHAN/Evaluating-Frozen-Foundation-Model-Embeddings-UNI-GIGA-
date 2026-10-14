import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Standard Attention-Based MIL pooling.

    Input:
        x : [N, 512]

    Returns:
        slide_embedding : [1, 512]
        attention_weights : [N, 1]
    """

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args
        ----
        x : Tensor
            Shape [N,512]

        Returns
        -------
        slide_embedding : Tensor
            Shape [1,512]

        attention_weights : Tensor
            Shape [N,1]
        """

        # Raw attention scores
        scores = self.attention(x)          # [N,1]

        # Normalize across patches
        attention = torch.softmax(scores, dim=0)   # [N,1]

        # Weighted average of patch embeddings
        slide_embedding = torch.sum(attention * x, dim=0, keepdim=True)

        return slide_embedding, attention