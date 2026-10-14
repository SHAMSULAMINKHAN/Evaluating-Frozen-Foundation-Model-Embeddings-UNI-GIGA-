import os
import csv
import argparse

import torch
from torch.utils.data import DataLoader

from datasets.survival_dataset import SurvivalDataset
from models.abmil_survival import ABMILSurvival

from losses.cox import CoxPHLoss
from metrics.cindex import concordance_index


@torch.no_grad()
def test(
    model,
    loader,
    device,
    output_csv
):

    model.eval()

    risks = []
    survivals = []
    events = []

    predictions = []

    # ======================================================
    # Generate predictions
    # ======================================================

    for batch in loader:

        features = (
            batch["features"]
            .squeeze(0)
            .to(device)
        )

        risk, _ = model(features)

        risk = risk.squeeze().cpu().item()

        survival = batch[
            "survival_months"
        ].item()

        event = batch[
            "event"
        ].item()

        risks.append(risk)
        survivals.append(survival)
        events.append(event)

        predictions.append(
            [
                batch["slide_id"][0],
                batch["patient_id"][0],
                survival,
                event,
                risk
            ]
        )

    # ======================================================
    # Convert to tensors
    # ======================================================

    risks_tensor = torch.tensor(
        risks,
        dtype=torch.float32
    )

    survivals_tensor = torch.tensor(
        survivals,
        dtype=torch.float32
    )

    events_tensor = torch.tensor(
        events,
        dtype=torch.float32
    )

    # ======================================================
    # Cox loss
    # ======================================================

    cox_loss_fn = CoxPHLoss()

    cox_loss = cox_loss_fn(
        risks_tensor,
        survivals_tensor,
        events_tensor
    )

    cox_loss_value = cox_loss.item()

    # ======================================================
    # C-index
    # ======================================================

    cindex = concordance_index(
        survivals,
        events,
        risks
    )

    # ======================================================
    # Print results
    # ======================================================

    print("=" * 60)

    print("Test Cox Loss :", cox_loss_value)

    print("Test C-index  :", cindex)

    print("=" * 60)

    # ======================================================
    # Save predictions + metrics
    # ======================================================

    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "slide_id",
                "patient_id",
                "survival_months",
                "event",
                "risk_score",
                "Cox loss",
                "C-index"
            ]
        )

        for prediction in predictions:

            writer.writerow(
                prediction
                + [
                    cox_loss_value,
                    cindex
                ]
            )

    print(
        "Saved:",
        output_csv
    )

    return cox_loss_value, cindex


def main(args):

    # ======================================================
    # Device
    # ======================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ======================================================
    # Test dataset
    # ======================================================

    test_dataset = SurvivalDataset(
        args.test_csv
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print(
        "Test slides:",
        len(test_dataset)
    )

    # ======================================================
    # Model
    # ======================================================

    model = ABMILSurvival(
        input_dim=args.feature_dim
    )

    checkpoint = torch.load(
        args.model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    model.eval()

    print(
        "Loaded model:",
        args.model_path
    )

    # ======================================================
    # Test
    # ======================================================

    test(
        model,
        test_loader,
        device,
        args.output_csv
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv",
        default="D:/datasets/completegiga/LUSC/MIL/splits/test.csv"
    )

    parser.add_argument(
        "--model_path",
        default="D:/datasets/completegiga/LUSC/MIL/abmail/best_model.pt"
    )

    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1536
    )

    parser.add_argument(
        "--output_csv",
        default=(
            "D:/datasets/completegiga/LUSC/MIL/abmail/"
            "test_predictions.csv"
        )
    )

    args = parser.parse_args()

    main(args)