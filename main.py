import torch
from torchvision import models

from pipeline import Pipeline
from configs import NUM_CLASSES, MODEL_NAMES

for model_name in MODEL_NAMES:
    print(f"Experiments for {model_name} have started!")

    model = getattr(models, model_name)()
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES)

    pilepine = Pipeline(model=model, name=model_name)
    pilepine.loop()