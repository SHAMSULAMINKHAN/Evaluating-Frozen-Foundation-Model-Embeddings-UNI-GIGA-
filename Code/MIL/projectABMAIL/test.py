import os
import csv
import argparse

import torch
from torch.utils.data import DataLoader


from datasets.survival_dataset import SurvivalDataset
from models.abmil_survival import ABMILSurvival

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


    for batch in loader:


        features = batch["features"].squeeze(0).to(device)


        risk, _ = model(features)


        risk = risk.squeeze().cpu().item()


        survival = batch["survival_months"].item()

        event = batch["event"].item()



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



    cindex = concordance_index(
        survivals,
        events,
        risks
    )


    print("=" * 60)

    print("Test C-index:", cindex)

    print("=" * 60)



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
                "risk_score"
            ]
        )


        writer.writerows(predictions)



    print(
        "Saved:",
        output_csv
    )


    return cindex





def main(args):


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("Device:", device)



    # Test dataset

    test_dataset = SurvivalDataset(
        args.test_csv
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )



    # Model

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
        default="D:/datasets/completeuni/BRCA/MIL/splits/test.csv"
    )


    parser.add_argument(
        "--model_path",
        default="D:/datasets/completeuni/BRCA/MIL/abmail/best_model.pt"
    )


    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1536
    )


    parser.add_argument(
        "--output_csv",
        default="D:/datasets/completeuni/BRCA/MIL/abmail/test_predictions.csv"
    )


    args = parser.parse_args()


    main(args)