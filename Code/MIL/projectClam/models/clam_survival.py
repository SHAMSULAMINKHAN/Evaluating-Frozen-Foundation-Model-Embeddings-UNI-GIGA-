import torch
import torch.nn as nn
import torch.nn.functional as F


from models.feature_projection import FeatureProjection
from models.gated_attention import GatedAttention
from models.survival_head import SurvivalHead



class CLAMSurvival(nn.Module):
    """
    CLAM-based survival model.

    Pipeline:

        Patch features
              |
        Projection
              |
        Gated Attention
              |
        Attention pooling
              |
        Survival head
              |
        Risk score

    """


    def __init__(
        self,
        input_dim=1536,
        hidden_dim=512,
        attention_dim=256,
        dropout=0.25
    ):

        super().__init__()


        self.projection = FeatureProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )


        self.attention = GatedAttention(
            input_dim=hidden_dim,
            attention_dim=attention_dim
        )


        self.survival_head = SurvivalHead(
            input_dim=hidden_dim,
            dropout=dropout
        )



    def forward(self, x):
        """
        Args:

            x:
                Patch embeddings

                Shape:
                [num_patches,input_dim]


        Returns:

            risk:
                Slide risk score

            attention:
                Patch importance weights

            slide_feature:
                Slide representation

        """


        # Project patch features

        H = self.projection(x)


        # Compute CLAM attention logits

        A = self.attention(H)


        # Normalize attention

        A = F.softmax(
            A,
            dim=1
        )


        # Weighted pooling

        slide_feature = torch.mm(
            A,
            H
        )


        # Predict survival risk

        risk = self.survival_head(
            slide_feature
        )


        return (
            risk,
            A,
            slide_feature
        )