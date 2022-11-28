import msgpack
import datetime
import torch
from matplotlib import pyplot as plt
from configs import FIGS_PATH

def log(filepath: str, object: any) -> None:
        with open(filepath, "wb") as outfile:
            packed = msgpack.packb(object)
            outfile.write(packed)

def get_logs(filepath: str) -> bytes:
    with open(filepath, "rb") as data_file:
        byte_data = data_file.read()
    return byte_data

class Logger():
    def __init__(self, process_name: str, info: any) -> None:
        self.info = info
        self.process_name = process_name
        self.best_entloss = 9999999.

    def checkpoint(self, model, logs_path: str, weights_path: str):
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%H:%M:%S")

        current_day = datetime.datetime.today()
        current_day = current_day.strftime("%d %m 20%y")

        log(filepath=f"{logs_path}{self.process_name}_logs", object=self.info)

        l = len(self.info['TEST_LOSS'])
        if (self.info['TEST_LOSS'][l-1] < self.best_entloss):
            torch.save(model, open(f"{weights_path}{self.process_name}", 'wb'))

        plt.plot(self.info['TEST_LOSS'])
        plt.plot(self.info['TRAIN_LOSS'])
        plt.legend(['тест', 'обучение'])
        plt.title(self.process_name)
        plt.savefig(FIGS_PATH + self.process_name)
        plt.figure().clear()

        plt.plot(self.info['TEST_ACC'])
        plt.plot(self.info['TRAIN_ACC'])
        plt.legend(['тест', 'обучение'])
        plt.title(self.process_name + " точность (accuracy)")
        plt.savefig(FIGS_PATH + "_acc_" + self.process_name)
        plt.figure().clear()

        print(f"\n-- checkpoint for {self.process_name} made day: {current_day} time: {current_time}")