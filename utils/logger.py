import msgpack
import datetime
import torch

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

    def checkpoint(self, model, logs_path: str, weights_path: str):
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%H:%M:%S")

        current_day = datetime.datetime.today()
        current_day = current_day.strftime("%d %m 20%y")

        log(filepath=f"{logs_path}{self.process_name}_logs", object=self.info)
        torch.save(model, open(f"{weights_path}{self.process_name}", 'wb'))
        print(f"\n-- checkpoint for {self.process_name} made day: {current_day} time: {current_time}")