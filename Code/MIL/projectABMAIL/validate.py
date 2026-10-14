import os
import csv
import argparse

import torch
from torch.utils.data import DataLoader


from datasets.survival_dataset import SurvivalDataset
from models.abmil_survival import ABMILSurvival
from metrics.cindex import concordance_index



@torch.no_grad()
def validate(
    model,
    loader,
    device,
    output_csv,
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
                risk,
            ]
        )



    cindex = concordance_index(
        survivals,
        events,
        risks
    )


    print("\nValidation C-index:", cindex)



    # Save predictions

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



    dataset = SurvivalDataset(
        args.val_csv
    )


    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )



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



    validate(
        model,
        loader,
        device,
        args.output_csv
    )





if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--val_csv",
        default="/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/splits/val.csv"
    )


    parser.add_argument(
        "--model_path",
        default="/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/abmial/outputs/best_model.pt"
    )


    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1536
    )


    parser.add_argument(
        "--output_csv",
        default="/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/abmial/outputs/validation_predictions.csv"
    )


    args = parser.parse_args()


    main(args)