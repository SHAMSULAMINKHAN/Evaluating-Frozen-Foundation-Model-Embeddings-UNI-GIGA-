import torch
import torch.nn as nn



class SurvivalHead(nn.Module):
    """
    Survival prediction head.

    Converts slide-level feature representation
    into a single risk score.

    Input:
        [batch, feature_dim]

    Output:
        [batch,1]
    """


    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        dropout=0.25
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),


            nn.ReLU(),


            nn.Dropout(
                dropout
            ),


            nn.Linear(
                hidden_dim,
                1
            )

        )



    def forward(self, x):
        """
        Args:
            x:
                Slide representation

                Shape:
                [batch,512]


        Returns:

            risk score

                Shape:
                [batch,1]
        """


        risk = self.network(x)


        return risk