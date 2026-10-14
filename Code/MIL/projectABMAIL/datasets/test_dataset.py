from torch.utils.data import DataLoader

from survival_dataset import SurvivalDataset


CSV_PATH = "D:/datasets/MIL/splits/train.csv"


def main():

    dataset = SurvivalDataset(CSV_PATH)

    print("=" * 60)
    print("Dataset size:", len(dataset))
    print("=" * 60)

    sample = dataset[0]

    print("Slide ID:", sample["slide_id"])
    print("Patient ID:", sample["patient_id"])

    print("Features shape:", sample["features"].shape)

    print("Survival:", sample["survival_months"].item())

    print("Event:", sample["event"].item())

    print("=" * 60)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    for batch in loader:

        print("One batch loaded successfully")

        print("Slide:", batch["slide_id"][0])

        print("Features:", batch["features"].shape)

        print("Survival:", batch["survival_months"].item())

        print("Event:", batch["event"].item())

        break


if __name__ == "__main__":
    main()