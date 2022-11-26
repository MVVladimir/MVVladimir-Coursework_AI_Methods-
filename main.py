from pipeline import Pipeline
from configs import MODEL_NAMES
from utils import model_loaders

for model_name in MODEL_NAMES:
    print(f"Experiments for {model_name} have started!")

    loader = getattr(model_loaders, f"{model_name}_loader")
    model = loader()

    pilepine = Pipeline(model=model, name=model_name)
    pilepine.loop()