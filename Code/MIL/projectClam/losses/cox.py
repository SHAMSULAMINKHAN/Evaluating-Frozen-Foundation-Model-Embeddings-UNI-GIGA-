import torch



def cox_ph_loss(
    risk,
    time,
    event
):
    """
    Cox proportional hazards partial likelihood loss.


    Args:

        risk:
            Predicted risk scores

            Shape:
            [N,1]


        time:
            Survival times

            Shape:
            [N]


        event:
            Event indicator

            1 = event occurred
            0 = censored


            Shape:
            [N]


    Returns:

        Cox loss

    """



    # Flatten risk

    risk = risk.reshape(-1)



    # Sort patients by survival time
    # longest -> shortest

    order = torch.argsort(
        time,
        descending=True
    )


    risk = risk[order]

    event = event[order]



    # Compute log cumulative hazard

    log_cumulative_risk = torch.logcumsumexp(
        risk,
        dim=0
    )



    # Partial likelihood

    losses = (
        risk - log_cumulative_risk
    ) * event



    # Negative log likelihood

    loss = -torch.sum(
        losses
    )


    # Normalize by number of events

    number_events = event.sum()


    if number_events > 0:

        loss = loss / number_events



    return loss