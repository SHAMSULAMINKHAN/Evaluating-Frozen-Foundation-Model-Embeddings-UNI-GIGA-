import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics.cindex import concordance_index


import torch

from datasets.survival_dataset import SurvivalDataset
from models.abmil_survival import ABMILSurvival


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load dataset
    dataset = SurvivalDataset(
        "/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/splits/val.csv"
    )

    sample = dataset[0]

    features = sample["features"].to(device)

    print("=" * 60)
    print("Slide:", sample["slide_id"])
    print("Patient:", sample["patient_id"])
    print("Feature shape:", features.shape)

    # Load model
    model = ABMILSurvival(
        input_dim=1536
    ).to(device)

    checkpoint = torch.load(
        "/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/abmial/outputs/best_model.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():

        risk, attention = model(features)

    print("=" * 60)
    print("Risk shape      :", risk.shape)
    print("Risk score      :", risk.item())

    print("Attention shape :", attention.shape)
    print("Attention sum   :", attention.sum().item())
    print("Attention min   :", attention.min().item())
    print("Attention max   :", attention.max().item())

    print("=" * 60)
    print("Inference successful.")


if __name__ == "__main__":
    main()