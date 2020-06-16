import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

# Creates a constructor for the DataReader class to create Pandas DataFrame
# objects based off of the dataset's file extension


class DataReader():
    def __init__(self, filepath):
        self.filepath = filepath

    # Retrieves the dataset's size in MB
    def retrieve_file_size(self):
        file_stats = os.stat(self.filepath)
        return (file_stats.st_size / (1024 ** 2))

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
    
    # Returns a list of available GPUs on the current device
    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [device.name for device in local_device_protos if device.device_type == 'GPU']
    
    # Checks if the current device has a GPU
    def is_gpu_available(self):
        return tf.test.gpu_device_name() != ''
    
    def trim (self, trimming, trim_ratio):
        if self.retrieve_file_size() >= 0.5:
            if trimming == 'random':
                trimmed_df = self.data_generator().sample(frac=trim_ratio)
                return trimmed_df
            elif trimming == 'stratified':
                return None # ADD STRATIFIED TRIMMING TECHNIQUE
        else:
            return None

    def trim_gpu (self, trim_option, trim_ratio):
        if self.is_gpu_available() == True:
            return self.data_generator()
        else:
            if trim_option != 'random' or trim_option != 'stratified':
                return trim('stratified', 0.10)
            else:
                return trim(trim_option, trim_ratio)