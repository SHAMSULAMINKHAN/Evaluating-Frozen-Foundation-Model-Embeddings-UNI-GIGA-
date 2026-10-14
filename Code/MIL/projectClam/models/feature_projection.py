import torch
import torch.nn as nn



class FeatureProjection(nn.Module):
    """
    Projection layer for foundation model embeddings.

    Converts:
        input_dim -> hidden_dim

    Example:
        UNI:
            1536 -> 512
    """


    def __init__(
        self,
        input_dim=1536,
        hidden_dim=512,
        dropout=0.25
    ):

        super().__init__()


        self.projection = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            )
        )



    def forward(self, x):
        """
        Args:
            x:
                Patch features

                Shape:
                [num_patches, input_dim]

        Returns:
                Projected features

                Shape:
                [num_patches, hidden_dim]
        """


        x = self.projection(x)

        return x