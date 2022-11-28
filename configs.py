from collections import namedtuple

NUM_CLASSES = 151
DATA_DIR = "./animal-151/dataset"
DEVICE = "cuda"
LOGS_PATH = "./logs/"
WEIGHTS_PATH = "./weights/"
TEST_IMGS = ["cat.png", "NorthernFlicker.png", "fox.png"]
TEST_IMGS_SAVE = "./pictures/"
FIGS_PATH = "./figs/"

DEFAULT_EPOCHS = 2
DEFAULT_BATCH = 16

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

shufflenet_v2_x0_5 = Config(DEFAULT_BATCH, 0.01, DEFAULT_EPOCHS)

resnet18 = Config(DEFAULT_BATCH, 0.1, DEFAULT_EPOCHS)
resnet34 = Config(DEFAULT_BATCH, 0.1, DEFAULT_EPOCHS)
resnet50 = Config(DEFAULT_BATCH, 0.1, DEFAULT_EPOCHS)

alexnet = Config(DEFAULT_BATCH, 0.1587, DEFAULT_EPOCHS)

mobilenet_v3_small = Config(DEFAULT_BATCH, 0.045, DEFAULT_EPOCHS)
mobilenet_v3_large = Config(DEFAULT_BATCH, 0.045, DEFAULT_EPOCHS)

efficientnet_v2_s = Config(DEFAULT_BATCH, 0.01, DEFAULT_EPOCHS)
efficientnet_v2_m = Config(DEFAULT_BATCH, 0.01, DEFAULT_EPOCHS)

vgg11 = Config(DEFAULT_BATCH, 0.01, DEFAULT_EPOCHS)

models_configs = ModelsConfigs(
                                shufflenet_v2_x0_5,
                                resnet18, resnet34, resnet50,
                                alexnet,
                                mobilenet_v3_small, mobilenet_v3_large,
                                efficientnet_v2_s, efficientnet_v2_m,
                                vgg11
                             )