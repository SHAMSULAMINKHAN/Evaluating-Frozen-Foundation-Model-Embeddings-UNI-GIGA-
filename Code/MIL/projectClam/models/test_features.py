import torch

from clam_survival import CLAMSurvival



# simulate one WSI

features = torch.randn(
    3924,
    1536
)



model = CLAMSurvival()



risk, attention, slide_feature = model(
    features
)



print(
    "Features:",
    features.shape
)


print(
    "Attention:",
    attention.shape
)


print(
    "Slide feature:",
    slide_feature.shape
)


print(
    "Risk:",
    risk.shape
)