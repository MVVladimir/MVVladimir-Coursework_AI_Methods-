from collections import namedtuple

NUM_CLASSES = 151
DATA_DIR = "./animal-151/dataset"
DEVICE = "cuda"
LOGS_PATH = "./logs/"
WEIGHTS_PATH = "./weights/"
MODEL_NAMES = ['shufflenet_v2_x0_5', 
               'resnet18', 'resnet34', 'resnet50',
               'alexnet', 
               'mobilenet_v3_small', 'mobilenet_v3_large', 
               'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']

print(MODEL_NAMES)

Config = namedtuple("Config", ["BATCH_SIZE", "LR", "EPOCHS"])
ModelsConfigs = namedtuple("ModelsConfigs", MODEL_NAMES)

shufflenet_v2_x0_5 = Config(4, 0.01, 2)

resnet18 = Config(4, 0.01, 2)
resnet34 = Config(4, 0.01, 2)
resnet50 = Config(4, 0.01, 2)

alexnet = Config(4, 0.01, 2)

mobilenet_v3_small = Config(4, 0.01, 2)
mobilenet_v3_large = Config(4, 0.01, 2)

efficientnet_v2_s = Config(4, 0.01, 2)
efficientnet_v2_m = Config(4, 0.01, 2)
efficientnet_v2_l = Config(4, 0.01, 2)

models_configs = ModelsConfigs(shufflenet_v2_x0_5, 
                                resnet18, resnet34, resnet50, 
                                alexnet, 
                                mobilenet_v3_small, mobilenet_v3_large, 
                                efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l) 