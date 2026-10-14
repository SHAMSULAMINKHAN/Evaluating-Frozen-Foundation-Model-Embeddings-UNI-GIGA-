import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.survival_dataset import SurvivalDataset
from models.abmil_survival import ABMILSurvival


CSV_PATH = "/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/splits/train.csv"


def main():

    dataset = SurvivalDataset(CSV_PATH)

    sample = dataset[0]

    features = sample["features"]

    model = ABMILSurvival()

    risk, attention = model(features)

    print("=" * 60)

    print("Input features :", features.shape)

    print("Risk shape     :", risk.shape)

    print("Risk score     :", risk.item())

    print("Attention shape:", attention.shape)

    print("Attention sum  :", attention.sum().item())

    print("Attention min  :", attention.min().item())

    print("Attention max  :", attention.max().item())

    print("=" * 60)


if __name__ == "__main__":
    main()