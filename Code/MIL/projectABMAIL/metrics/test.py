import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics.cindex import concordance_index


def main():

    survival = [
        10,
        50,
        5,
        30
    ]

    events = [
        1,
        0,
        1,
        1
    ]

    risk = [
        0.8,
        -0.2,
        1.4,
        0.1
    ]


    cindex = concordance_index(
        survival,
        events,
        risk
    )


    print("C-index:")
    print(cindex)


if __name__ == "__main__":
    main()