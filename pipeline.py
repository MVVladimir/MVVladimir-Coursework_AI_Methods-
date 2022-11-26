import torch
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from torchcam.methods import XGradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import transforms
from matplotlib import pyplot as plt

from statistics import mean

from configs import models_configs
from configs import DEVICE, DATA_DIR, LOGS_PATH, WEIGHTS_PATH, TEST_IMGS, TEST_IMGS_SAVE
from utils.dataloaders import get_data_loaders
from utils.logger import Logger
from utils import model_loaders

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Pipeline():
    def __init__(self, model, name: str) -> None:
        self.name = name
        self.model = model
        self.device = torch.device(DEVICE)
        self.model.to(self.device)

        self.config = getattr(models_configs, name)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LR)
        self.results = self.config._asdict()
        self.logger = Logger(process_name=name, info=self.results)
        self.logger.info['TEST_LOSS'] = []
        self.logger.info['TRAIN_LOSS'] = []

    def train(self, model, criterion, x, y):
        pred = model(x.to(self.device))
        loss = criterion(pred, y.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu()

    def test(self, model, criterion, x, y):
        with torch.no_grad():
            pred = model(x.to(self.device))
            loss = criterion(pred, y.to(self.device))
        return loss.cpu()

    def make_test_image(self, epoch):
        for img_path in TEST_IMGS:
            image = Image.open(img_path)
            image = image.convert()

            get_layers = getattr(model_loaders, f"{self.name}_params")
            conv = get_layers(self.model)

            cam = XGradCAM(self.model, conv)

            input = transforms.PILToTensor()(image)
            input = input.to(dtype=torch.float)
            input = input.to(device=self.device)

            self.model.train()
            self.model.zero_grad()

            out = self.model(input.unsqueeze(0))
            out = out.cpu()

            activation_map = cam(out.squeeze(0).argmax().item(), out)
            result = overlay_mask(to_pil_image(input), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

            plt.imshow(result);
            plt.tight_layout();
            plt.savefig(TEST_IMGS_SAVE + self.name + "_epoch_" + str(epoch) + ".png")
            plt.figure().clear()

            cam.remove_hooks()
            del cam
            del activation_map

            self.model.zero_grad()
            self.model.eval()

    def loop(self):

        self.make_test_image(0)

        for epoch in range(self.config.EPOCHS):
            print(f"\nEPOCH {epoch+1} OUT OF {self.config.EPOCHS}")

            train_loader, validation_loader, test_loader = get_data_loaders(data_dir=DATA_DIR, batch_size=self.config.BATCH_SIZE)

            train_loss = []
            test_loss = []

            self.model.train()
            for batch in tqdm(train_loader):
                data, labels = batch
                train_loss.append(self.train(model=self.model, criterion=self.criterion, x=data, y=labels).item())

            self.model.eval()
            for batch in tqdm(test_loader):
                data, labels = batch
                test_loss.append(self.test(model=self.model, criterion=self.criterion, x=data, y=labels).item())

            for batch in tqdm(validation_loader): # we do not tune hyperparameters !
                data, labels = batch
                test_loss.append(self.test(model=self.model, criterion=self.criterion, x=data, y=labels).item())

            self.logger.info['TRAIN_LOSS'].append(mean(train_loss))
            self.logger.info['TEST_LOSS'].append(mean(test_loss))
            self.logger.checkpoint(self.model, logs_path=LOGS_PATH, weights_path=WEIGHTS_PATH)

            self.make_test_image(epoch)
