from collections import namedtuple

NUM_CLASSES = 151
DATA_DIR = "./animal-151/dataset"
DEVICE = "cuda"
LOGS_PATH = "./logs/"
WEIGHTS_PATH = "./weights/"
MODEL_NAMES = ['shufflenet_v2_x0_5', 'resnet50']

Config = namedtuple("Config", ["BATCH_SIZE", "LR", "EPOCHS"])
ModelsConfigs = namedtuple("ModelsConfigs", MODEL_NAMES)

config_shuffleNetV2X0_5_config = Config(4, 0.01, 2)
resnet50_config = Config(4, 0.01, 2)

models_configs = ModelsConfigs(config_shuffleNetV2X0_5_config, resnet50_config)