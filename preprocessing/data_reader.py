import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

# Creates a constructor for the DataReader class to create Pandas DataFrame
# objects based off of the dataset's file extension


class DataReader():
    def __init__(self, filepath, trim, trim_ratio):
        self.filepath = filepath
        self.trim = trim
        self.trim_ratio = trim_ratio

    # Sets the trim_ratio to the requested value
    def set_trim_ratio(self, new_trim_ratio):
        self.trim_ratio = new_trim_ratio

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
    
    # Trims the dataset based off of the technique specified and the ratio of data to be trimmed specified
    def trim (self):
        if self.retrieve_file_size() >= 0.5:
            if self.trim == True:
                trimmed_df = self.data_generator().sample(frac=self.trim_ratio)
                return trimmed_df
            elif self.trim == False:
                return self.data_generator()
        else:
            return "The file size is too small trim!".upper()

    def trim_gpu (self):
        if self.is_gpu_available() == True:
            return self.data_generator()
        else:
            if self.trim != True and self.trim != False:
                self.set_trim_ratio(0.20)
                return trim()
            else:
                return trim()