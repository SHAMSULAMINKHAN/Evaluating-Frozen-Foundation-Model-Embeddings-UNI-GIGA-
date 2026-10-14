
from survival_dataset import SurvivalWSIDataset


CSV = "D:/datasets/MIL/splits/train.csv"


dataset = SurvivalWSIDataset(CSV)


print(
    "Number of slides:",
    len(dataset)
)


sample = dataset[0]


print(sample.keys())


print(
    "Feature shape:",
    sample["features"].shape
)


print(
    "Survival:",
    sample["time"]
)


print(
    "Event:",
    sample["event"]
)


print(
    "Slide:",
    sample["slide_id"]
)