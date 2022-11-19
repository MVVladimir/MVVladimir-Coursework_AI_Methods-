import torch
import numpy as np
import random
from torchvision.models import shufflenet_v2_x0_5
from tqdm import tqdm

from statistics import mean

from configs import ModelsConfigs
from configs import DEVICE, DATA_DIR, LOGS_PATH, WEIGHTS_PATH
from utils.dataloaders import get_data_loaders
from utils.logger import Logger

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Pipeline():
    def __init__(self, model, name: str) -> None:
        self.model = model
        self.device = torch.device(DEVICE)
        self.model.to(self.device)

        self.config = getattr(ModelsConfigs, name)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LR)
        self.results = self.config._asdict()
        self.logger = Logger(process_name="shuffleNetV2X0_5", info=self.results)
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

    def loop(self):
        train_loader, validation_loader, test_loader = get_data_loaders(data_dir=DATA_DIR, batch_size=self.config.BATCH_SIZE)

        for epoch in range(self.config.BATCH_SIZE):
            print(f"\nEPOCH {epoch+1} OUT OF {self.config.EPOCHS}")

            train_loss = []
            test_loss = []

            for batch in tqdm(train_loader):
                data, labels = batch
                train_loss.append(self.train(model=self.model, criterion=self.criterion, x=data, y=labels).item())
            
            for batch in tqdm(test_loader):
                data, labels = batch
                test_loss.append(self.test(model=self.model, criterion=self.criterion, x=data, y=labels).item())

            for batch in tqdm(validation_loader): # we do not tune hyperparameters !
                data, labels = batch
                test_loss.append(self.test(model=self.model, criterion=self.criterion, x=data, y=labels).item())

            self.logger.info['TRAIN_LOSS'].append(mean(train_loss))
            self.logger.info['TEST_LOSS'].append(mean(test_loss))
            self.logger.checkpoint(self.model, logs_path=LOGS_PATH, weights_path=WEIGHTS_PATH)

# pipeline = Pipeline()
# pipeline.loop()