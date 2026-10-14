import torch
import torch.nn as nn

from models.feature_projection import FeatureProjection
from models.attention_pooling import AttentionPooling
from models.survival_head import SurvivalHead


class ABMILSurvival(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL)
    for Cox survival prediction.

    Input:
        [N,1536]

    Output:
        risk_score : [1,1]
        attention_weights : [N,1]
    """

    def __init__(
        self,
        input_dim=1536,
        projection_dim=512,
        attention_dim=256,
        dropout=0.25,
    ):
        super().__init__()

        self.feature_projection = FeatureProjection(
            input_dim=input_dim,
            output_dim=projection_dim,
            dropout=dropout,
        )

        self.attention_pooling = AttentionPooling(
            input_dim=projection_dim,
            hidden_dim=attention_dim,
        )

        self.survival_head = SurvivalHead(
            input_dim=projection_dim,
            hidden_dim=attention_dim,
            dropout=dropout,
        )

    def forward(self, features):
        """
        Parameters
        ----------
        features : Tensor
            Shape [N,1536]

        Returns
        -------
        risk_score : Tensor
            Shape [1,1]

        attention_weights : Tensor
            Shape [N,1]
        """

        projected_features = self.feature_projection(features)

        slide_embedding, attention_weights = self.attention_pooling(
            projected_features
        )

        risk_score = self.survival_head(slide_embedding)

        return risk_score, attention_weights