import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast, GradScaler

from datasets.survival_dataset import SurvivalWSIDataset

from models.clam_survival import CLAMSurvival

from losses.cox import cox_ph_loss

from metrics.cindex import compute_cindex



# ============================
# Configuration
# ============================

TRAIN_CSV = (
    "D:/datasets/completeuni/LUAD/MIL/splits/train.csv"
)

VAL_CSV = (
    "D:/datasets/completeuni/LUAD/MIL/splits/val.csv"
)


SAVE_DIR = (
    "D:/datasets/completeuni/LUAD/MIL/clam"
)


CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "best_clam_survival.pt"
)


HISTORY_PATH = os.path.join(
    SAVE_DIR,
    "training_history.csv"
)



EPOCHS = 100

LR = 1e-4

WEIGHT_DECAY = 1e-5


PATIENCE = 15



DEVICE = torch.device(
   # "cuda"
    #if torch.cuda.is_available()
    #else 
    "cpu"
    
)


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


print(
    "Device:",
    DEVICE
)



# ============================
# Train epoch
# ============================

def train_epoch(
    model,
    loader,
    optimizer,
    scaler
):

    model.train()


    risks = []
    times = []
    events = []



    for batch in loader:


        features = batch["features"][0]


        features = features.to(
            DEVICE,
            non_blocking=True
        )


        with autocast():


            risk, _, _ = model(
                features
            )


        risks.append(
            risk
        )


        times.append(
            batch["time"]
        )


        events.append(
            batch["event"]
        )



    risks = torch.cat(
        risks,
        dim=0
    )


    times = torch.cat(
        times,
        dim=0
    ).to(DEVICE)


    events = torch.cat(
        events,
        dim=0
    ).to(DEVICE)



    loss = cox_ph_loss(
        risks,
        times,
        events
    )



    optimizer.zero_grad()


    scaler.scale(
        loss
    ).backward()


    scaler.step(
        optimizer
    )


    scaler.update()



    return loss.item()



# ============================
# Validation
# ============================

def validate(
    model,
    loader
):

    model.eval()


    risks = []
    times = []
    events = []



    with torch.no_grad():


        for batch in loader:


            features = batch["features"][0]


            features = features.to(
                DEVICE
            )


            with autocast():


                risk, _, _ = model(
                    features
                )


            risks.append(
                risk.cpu()
            )


            times.append(
                batch["time"]
            )


            events.append(
                batch["event"]
            )



    risks = torch.cat(
        risks,
        dim=0
    )


    times = torch.cat(
        times,
        dim=0
    )


    events = torch.cat(
        events,
        dim=0
    )



    loss = cox_ph_loss(
        risks,
        times,
        events
    )


    cindex = compute_cindex(
        risks,
        times,
        events
    )


    return loss.item(), cindex



# ============================
# Dataset
# ============================

train_dataset = SurvivalWSIDataset(
    TRAIN_CSV
)


val_dataset = SurvivalWSIDataset(
    VAL_CSV
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



# ============================
# Model
# ============================

model = CLAMSurvival(
    input_dim=1536
)


model.to(
    DEVICE
)



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY

)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=5

)



scaler = GradScaler()



# ============================
# Training
# ============================

best_cindex = -1

early_stop_counter = 0


history = []



for epoch in range(EPOCHS):


    train_loss = train_epoch(

        model,

        train_loader,

        optimizer,

        scaler

    )


    val_loss, val_cindex = validate(

        model,

        val_loader

    )


    scheduler.step(
        val_cindex
    )


    current_lr = optimizer.param_groups[0]["lr"]



    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"C-index: {val_cindex:.4f} | "
        f"LR: {current_lr:.2e}"
    )



    history.append(

        {

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "cindex": val_cindex,

            "lr": current_lr

        }

    )


    pd.DataFrame(history).to_csv(

        HISTORY_PATH,

        index=False

    )



    # Save best model

    if val_cindex > best_cindex:


        best_cindex = val_cindex


        early_stop_counter = 0



        torch.save(

            {

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_cindex":
                    best_cindex

            },

            CHECKPOINT_PATH

        )


        print(
            "Saved best model"
        )



    else:


        early_stop_counter += 1



    if early_stop_counter >= PATIENCE:


        print(
            "Early stopping triggered"
        )


        break



print(
    "Training finished"
)


print(
    "Best C-index:",
    best_cindex
)