import numpy as np


def concordance_index(
    survival_times,
    events,
    risk_scores,
):
    """
    Compute Harrell's concordance index.

    Parameters
    ----------
    survival_times : array-like
        Observed survival times

    events : array-like
        1 = event occurred
        0 = censored

    risk_scores : array-like
        Predicted risk scores.
        Higher means higher risk.

    Returns
    -------
    c_index : float
    """

    survival_times = np.asarray(survival_times)
    events = np.asarray(events)
    risk_scores = np.asarray(risk_scores)


    concordant = 0
    permissible = 0


    n = len(survival_times)


    for i in range(n):

        for j in range(n):

            if i == j:
                continue


            # Patient i had observed event
            if events[i] == 1:

                # i survived less than j
                if survival_times[i] < survival_times[j]:

                    permissible += 1


                    # Higher risk should belong to shorter survival
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1

                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5


    if permissible == 0:
        return 0.0


    return concordant / permissible