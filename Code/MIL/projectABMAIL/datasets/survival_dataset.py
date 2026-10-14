import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    """
    Dataset for whole-slide survival prediction using pre-extracted
    foundation model features.

    Expected CSV columns:
        slide_id
        patient_id
        feature_path
        survival_months
        event

    Each feature file (.npy) should contain:
    {
        "features": numpy.ndarray (N_patches, feature_dim),
        "patch_names": list
    }
    """

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        required_columns = [
            "slide_id",
            "patient_id",
            "feature_path",
            "survival_months",
            "event",
        ]

        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        print(f"\nLoaded {len(self.data)} slides")
        print(f"CSV: {csv_file}")

        print("\nEvent distribution:")
        print(self.data["event"].value_counts().sort_index())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        feature_path = row["feature_path"]

        if not os.path.exists(feature_path):
            raise FileNotFoundError(feature_path)

        bag = np.load(feature_path, allow_pickle=True).item()

        features = bag["features"]

        features = torch.tensor(features, dtype=torch.float32)

        sample = {
            "slide_id": row["slide_id"],
            "patient_id": row["patient_id"],
            "features": features,
            "survival_months": torch.tensor(
                row["survival_months"],
                dtype=torch.float32
            ),
            "event": torch.tensor(
                row["event"],
                dtype=torch.float32
            )
        }

        return sample