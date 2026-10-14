import os
import csv
import random
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


from datasets.survival_dataset import SurvivalDataset
from models.abmil_survival import ABMILSurvival

from losses.cox import CoxPHLoss
from metrics.cindex import concordance_index



def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()


    risks = []
    survivals = []
    events = []


    for batch in loader:


        features = batch["features"].squeeze(0).to(device)

        survival = batch["survival_months"].to(device)

        event = batch["event"].to(device)



        risk, _ = model(features)



        risks.append(
            risk.squeeze()
        )

        survivals.append(
            survival.squeeze()
        )

        events.append(
            event.squeeze()
        )



    risks = torch.stack(risks)

    survivals = torch.stack(survivals)

    events = torch.stack(events)



    loss = criterion(
        risks,
        survivals,
        events
    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    return loss.item()



@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()


    risks = []
    survivals = []
    events = []


    for batch in loader:


        features = batch["features"].squeeze(0).to(device)


        risk, _ = model(features)



        risks.append(
            risk.item()
        )


        survivals.append(
            batch["survival_months"].item()
        )


        events.append(
            batch["event"].item()
        )



    cindex = concordance_index(
        survivals,
        events,
        risks
    )


    return cindex



@torch.no_grad()
def save_predictions(
    model,
    loader,
    device,
    output_file
):

    model.eval()


    rows = []


    for batch in loader:


        features = batch["features"].squeeze(0).to(device)


        risk, _ = model(features)


        rows.append(
            [
                batch["slide_id"][0],
                batch["patient_id"][0],
                batch["survival_months"].item(),
                batch["event"].item(),
                risk.item()
            ]
        )


    with open(
        output_file,
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


        writer.writerows(rows)




def main(args):


    seed_everything(args.seed)

    device = torch.device("cpu")



    #device = torch.device(

     #   "cuda"
      #  if torch.cuda.is_available()
       # else "cpu"
    #)


    print("Device:", device)



    train_dataset = SurvivalDataset(
        args.train_csv
    )


    val_dataset = SurvivalDataset(
        args.val_csv
    )



    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )



    model = ABMILSurvival(
        input_dim=args.feature_dim
    )


    model.to(device)



    criterion = CoxPHLoss()



    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )



    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10
    )



    os.makedirs(
        args.output_dir,
        exist_ok=True
    )



    history_file = os.path.join(
        args.output_dir,
        "training_history.csv"
    )


    best_model = os.path.join(
        args.output_dir,
        "best_model.pt"
    )


    prediction_file = os.path.join(
        args.output_dir,
        "validation_predictions.csv"
    )



    best_cindex = 0

    epochs_without_improvement = 0



    with open(history_file, "w") as f:


        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "loss",
                "cindex",
                "lr"
            ]
        )



        for epoch in range(args.epochs):


            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )



            cindex = evaluate(
                model,
                val_loader,
                device
            )



            scheduler.step(cindex)



            lr = optimizer.param_groups[0]["lr"]



            print(
                f"Epoch {epoch+1}/{args.epochs} "
                f"Loss:{loss:.4f} "
                f"C-index:{cindex:.4f} "
                f"LR:{lr:.2e}"
            )



            writer.writerow(
                [
                    epoch+1,
                    loss,
                    cindex,
                    lr
                ]
            )



            if cindex > best_cindex:


                best_cindex = cindex

                epochs_without_improvement = 0



                torch.save(
                    model.state_dict(),
                    best_model
                )


                save_predictions(
                    model,
                    val_loader,
                    device,
                    prediction_file
                )


                print("Saved best model")



            else:

                epochs_without_improvement += 1



            if epochs_without_improvement >= args.patience:

                print(
                    "Early stopping triggered"
                )

                break



    print("\nTraining finished")

    print(
        "Best C-index:",
        best_cindex
    )




if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--train_csv",
        default="D:/datasets/completeuni/LUAD/MIL/splits/train.csv"
    )


    parser.add_argument(
        "--val_csv",
        default="D:/datasets/completeuni/LUAD/MIL/splits/val.csv"
    )


    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1536
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )


    parser.add_argument(
        "--patience",
        type=int,
        default=30
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=123
    )


    parser.add_argument(
        "--output_dir",
        default="D:/datasets/completeuni/LUAD/MIL/abmil"
    )


    args = parser.parse_args()


    main(args)