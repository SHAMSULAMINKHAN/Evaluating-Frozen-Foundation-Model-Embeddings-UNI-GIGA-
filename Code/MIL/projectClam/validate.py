import os
import pandas as pd
import torch
from torch.utils.data import DataLoader


from datasets.survival_dataset import SurvivalWSIDataset

from models.clam_survival import CLAMSurvival

from losses.cox import cox_ph_loss

from metrics.cindex import compute_cindex



# ============================
# Configuration
# ============================

VAL_CSV = (
    "/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/splits/val.csv"
)


CHECKPOINT = (
    "/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/"
    "checkpoints/best_clam_survival.pt"
)


OUTPUT_FILE = (
    "/homes/suamin/Bioinformatics_project/code/UNI2/LUAD/"
    "validation_predictions.csv"
)



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



print(
    "Device:",
    DEVICE
)



# ============================
# Dataset
# ============================

dataset = SurvivalWSIDataset(
    VAL_CSV
)


loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)



print(
    "Validation slides:",
    len(dataset)
)



# ============================
# Load model
# ============================

model = CLAMSurvival(
    input_dim=1536
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(
    DEVICE
)


model.eval()



print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"]
)



# ============================
# Evaluation
# ============================

risks = []

times = []

events = []

slide_ids = []

patient_ids = []



with torch.no_grad():


    for batch in loader:


        features = batch["features"][0]


        features = features.to(
            DEVICE
        )


        risk, attention, slide_feature = model(
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


        slide_ids.append(
            batch["slide_id"][0]
        )


        patient_ids.append(
            batch["patient_id"][0]
        )



# Combine results

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



# ============================
# Metrics
# ============================


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



print(
    f"Validation Cox loss: {loss.item():.4f}"
)


print(
    f"Validation C-index: {cindex:.4f}"
)



# ============================
# Save predictions
# ============================


results = pd.DataFrame(
    {
        "slide_id": slide_ids,

        "patient_id": patient_ids,

        "survival_months":
            times.numpy(),

        "event":
            events.numpy(),

        "risk_score":
            risks.squeeze().numpy()
    }
)



results.to_csv(
    OUTPUT_FILE,
    index=False
)



print(
    "Saved predictions:"
)

print(
    OUTPUT_FILE
)