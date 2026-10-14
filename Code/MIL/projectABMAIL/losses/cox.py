import torch
import torch.nn as nn


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards negative partial log-likelihood.

    Inputs:
        risk_scores:
            Predicted log hazard ratios
            Shape: [N]

        survival_times:
            Survival months
            Shape: [N]

        events:
            1 = observed event
            0 = censored
            Shape: [N]

    Returns:
        Cox loss (scalar)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        risk_scores,
        survival_times,
        events,
    ):

        # Flatten tensors
        risk_scores = risk_scores.view(-1)
        survival_times = survival_times.view(-1)
        events = events.view(-1)

        # Sort patients by decreasing survival time
        order = torch.argsort(
            survival_times,
            descending=True
        )

        risk_scores = risk_scores[order]
        events = events[order]

        # Compute log cumulative hazard
        log_cumulative_hazard = torch.logcumsumexp(
            risk_scores,
            dim=0
        )

        # Partial log likelihood
        log_likelihood = (
            risk_scores - log_cumulative_hazard
        )

        # Only uncensored patients contribute
        loss = -torch.sum(
            log_likelihood * events
        )

        # Normalize by number of events
        num_events = torch.sum(events)

        if num_events > 0:
            loss = loss / num_events

        return loss