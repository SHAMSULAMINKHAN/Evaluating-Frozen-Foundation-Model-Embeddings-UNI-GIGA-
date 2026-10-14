import numpy as np

from lifelines.utils import concordance_index



def compute_cindex(
    risk_scores,
    survival_times,
    events
):
    """
    Calculate survival concordance index.


    Args:

        risk_scores:
            Model predicted risk

            Higher = higher risk


        survival_times:
            Observed survival time


        events:
            1 = death observed
            0 = censored


    Returns:

        C-index value
    """



    # Convert tensors to numpy

    if hasattr(risk_scores, "detach"):

        risk_scores = (
            risk_scores
            .detach()
            .cpu()
            .numpy()
        )


    if hasattr(survival_times, "detach"):

        survival_times = (
            survival_times
            .detach()
            .cpu()
            .numpy()
        )


    if hasattr(events, "detach"):

        events = (
            events
            .detach()
            .cpu()
            .numpy()
        )



    # lifelines expects:
    #
    # higher prediction = longer survival
    #
    # but our model:
    #
    # higher risk = shorter survival
    #
    # therefore use negative risk


    cindex = concordance_index(

        survival_times,

        -risk_scores,

        events

    )


    return cindex