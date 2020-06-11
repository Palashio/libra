import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

# import torch.cuda as cuda

# Creates a constructor for the DataReader class to create Pandas DataFrame
# objects based off of the dataset's file extension


class DataReader():
    def __init__(self, filepath):
        self.filepath = filepath

    # Retrieves the dataset's file extension
    def retrieve_extension(self):
        ext_index = self.filepath.rindex('.')
        return self.filepath[ext_index:]

    # Creates a Pandas DataFrame object based off of what the inputted
    # dataset's extension is
    def data_generator(self):
        if self.retrieve_extension() == '.csv':
            df = pd.read_csv(self.filepath)
        elif self.retrieve_extension() == '.xlsx':
            df = pd.read_excel(self.filepath)
            for data in df:
                if df[data].dtype.name == 'int64':
                    df[data] = df[data].astype(float)
        elif self.retrieve_extension() == '.json':
            df = pd.read_json(self.filepath)
        return df
    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    def is_gpu_available_1(self):
        return tf.test.gpu_device_name() != ''
    # def is_gpu_available_2(self):
    #     return cuda.is_available()