import torch

from cox import cox_ph_loss



# Example predictions

risk = torch.tensor(
    [
        [1.2],
        [0.5],
        [-0.3],
        [2.0]
    ]
)



# Survival months

time = torch.tensor(
    [
        10.0,
        30.0,
        50.0,
        5.0
    ]
)



# Events

event = torch.tensor(
    [
        1,
        1,
        0,
        1
    ],
    dtype=torch.float32
)



loss = cox_ph_loss(
    risk,
    time,
    event
)



print(
    "Cox loss:",
    loss
)