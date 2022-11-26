from collections import namedtuple

NUM_CLASSES = 151
DATA_DIR = "./animal-151/dataset"
DEVICE = "cuda"
LOGS_PATH = "./logs/"
WEIGHTS_PATH = "./weights/"
TEST_IMGS = ["./cat.png", ]
TEST_IMGS_SAVE = "./pictures/"
FIGS_PATH = "./figs/"

MODEL_NAMES = [
               'shufflenet_v2_x0_5',
               'resnet18', 'resnet34', 'resnet50',
               'alexnet',
               'mobilenet_v3_small', 'mobilenet_v3_large',
               'efficientnet_v2_s', 'efficientnet_v2_m',
               'vgg11'
]

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

vgg11 = Config(4, 0.01, 2)

models_configs = ModelsConfigs(
                                shufflenet_v2_x0_5,
                                resnet18, resnet34, resnet50,
                                alexnet,
                                mobilenet_v3_small, mobilenet_v3_large,
                                efficientnet_v2_s, efficientnet_v2_m,
                                vgg11)