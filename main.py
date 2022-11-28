from pipeline import Pipeline
from configs import MODEL_NAMES, models_configs
from utils import model_loaders

if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        print(f"Experiments for {model_name} have started!")

        config = getattr(models_configs, model_name)
        print(config.BATCH_SIZE)

        loader = getattr(model_loaders, f"{model_name}_loader")
        model = loader()

        pilepine = Pipeline(model=model, name=model_name)
        pilepine.loop()