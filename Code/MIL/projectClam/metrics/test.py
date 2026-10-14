import torch

from cindex import compute_cindex



risk = torch.tensor(
    [
        2.0,
        1.0,
        -1.0,
        0.5
    ]
)



time = torch.tensor(
    [
        10.0,
        30.0,
        80.0,
        50.0
    ]
)



event = torch.tensor(
    [
        1,
        1,
        0,
        1
    ]
)



cindex = compute_cindex(
    risk,
    time,
    event
)



print(
    "C-index:",
    cindex
)