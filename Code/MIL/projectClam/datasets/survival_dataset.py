import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset



class SurvivalWSIDataset(Dataset):
    """
    PyTorch Dataset for Whole Slide Image survival prediction.

    Each sample represents one WSI.

    Input:
        CSV file containing:
            slide_id
            patient_id
            feature_path
            survival_months
            event

    Output:
        {
            "features": Tensor [num_patches, feature_dim],
            "time": Tensor,
            "event": Tensor,
            "slide_id": str,
            "patient_id": str
        }
    """


    def __init__(self, csv_file):
        """
        Initialize dataset.

        Args:
            csv_file (str):
                Path to train/validation/test CSV
        """


        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"CSV file not found: {csv_file}"
            )


        self.data = pd.read_csv(csv_file)


        required_columns = [
            "slide_id",
            "patient_id",
            "feature_path",
            "survival_months",
            "event"
        ]


        for col in required_columns:

            if col not in self.data.columns:
                raise ValueError(
                    f"Missing column in CSV: {col}"
                )


        print(
            f"Loaded {len(self.data)} slides from {csv_file}"
        )



    def __len__(self):
        """
        Number of WSIs.
        """

        return len(self.data)



    def __getitem__(self, idx):
        """
        Load one WSI sample.

        Args:
            idx:
                Index of slide

        Returns:
            Dictionary containing:
                patch features
                survival information
        """


        row = self.data.iloc[idx]


        # -----------------------------
        # Load UNI feature file
        # -----------------------------

        feature_path = row["feature_path"]


        if not os.path.exists(feature_path):

            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )


        feature_data = np.load(
            feature_path,
            allow_pickle=True
        ).item()



        # Expected format:
        #
        # {
        #     "features": numpy array,
        #     "patch_names": list
        # }


        features = feature_data["features"]
        if features.ndim != 2:
            raise ValueError(
                f"Expected feature matrix [N,D], got {features.shape}"
            )



        # Convert numpy array to tensor

        features = torch.tensor(
            features,
            dtype=torch.float32
        )
        if features.shape[1] != 1536:
            raise ValueError(
                f"Expected UNI dimension 1536, got {features.shape[1]}"
            )



        # -----------------------------
        # Survival information
        # -----------------------------


        survival_time = torch.tensor(
            row["survival_months"],
            dtype=torch.float32
        )


        event = torch.tensor(
            row["event"],
            dtype=torch.float32
        )



        return {

            "features": features,

            "time": survival_time,

            "event": event,

            "slide_id": row["slide_id"],

            "patient_id": row["patient_id"]

        }