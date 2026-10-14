import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from losses.cox import CoxPHLoss


def main():

    loss_fn = CoxPHLoss()


    # Example batch
    risk_scores = torch.tensor(
        [
            0.8,
            -0.2,
            1.4,
            0.1
        ],
        dtype=torch.float32
    )


    survival_times = torch.tensor(
        [
            10,
            50,
            5,
            30
        ],
        dtype=torch.float32
    )


    events = torch.tensor(
        [
            1,
            0,
            1,
            1
        ],
        dtype=torch.float32
    )


    loss = loss_fn(
        risk_scores,
        survival_times,
        events
    )


    print("Risk scores:")
    print(risk_scores)

    print()

    print("Survival:")
    print(survival_times)

    print()

    print("Events:")
    print(events)

    print()

    print("Cox loss:")
    print(loss.item())


if __name__ == "__main__":
    main()