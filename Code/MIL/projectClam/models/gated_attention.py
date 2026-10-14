import torch
import torch.nn as nn



class GatedAttention(nn.Module):
    """
    Gated attention mechanism used in CLAM.

    Implements:

        A = W(tanh(Vx) * sigmoid(Ux))


    Input:
        N x feature_dim

    Output:
        1 x N attention scores
    """



    def __init__(
        self,
        input_dim=512,
        attention_dim=256
    ):

        super().__init__()



        # First attention branch

        self.V = nn.Sequential(

            nn.Linear(
                input_dim,
                attention_dim
            ),

            nn.Tanh()

        )



        # Gating branch

        self.U = nn.Sequential(

            nn.Linear(
                input_dim,
                attention_dim
            ),

            nn.Sigmoid()

        )



        # Final attention scorer

        self.attention_weights = nn.Linear(
            attention_dim,
            1
        )



    def forward(self, x):
        """
        Args:

            x:
                Patch features

                Shape:
                [num_patches, input_dim]


        Returns:

            attention_scores:

                Shape:
                [1,num_patches]
        """


        V = self.V(x)


        U = self.U(x)



        gated = V * U



        A = self.attention_weights(
            gated
        )


        # transpose:

        # before:
        # [num_patches,1]

        # after:
        # [1,num_patches]

        A = torch.transpose(
            A,
            1,
            0
        )


        return A