import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib

# Creates a constructor for the DataReader class to create Pandas DataFrame
# objects based off of the dataset's file extension, user's preference on 
# random trimming of the dataset, and the proportion of data to be trimmed 
# specified by the user
class DataReader():
    def __init__(self, filepath, trim=False, trim_format='random', trim_ratio=0.20):
        self.filepath = filepath
        self.trim = trim
        self.trim_format = trim_format
        self.trim_ratio = trim_ratio

        self.trim_gpu()


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

        if self.is_gpu_available() == False:
            self.trim = True
        if self.trim == True:
            if self.trim_format == 'random':
                df = df.sample(frac=(1.0 - self.trim_ratio))
            elif self.trim_format == 'stratify':
                # ADD STRATIFYING METHOD
                return None
        return df
    
    # Returns a list of available GPUs on the current device
    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [ device.name for device in local_device_protos 
                 if device.device_type == 'GPU']
    
    # Checks if the current device has a GPU
    def is_gpu_available(self):
        return tf.test.gpu_device_name() != ''
    
    # Trims the dataset based off of user preference and the ratio 
    # of data to be trimmed specified
    def random_trim (self):
        if self.trim == True:
            trimmed_df = self.data_generator().sample(frac=(1.0 - self.trim_ratio))
            return trimmed_df
        elif self.trim == False:
            return self.data_generator()
        
    # def stratified_strim(self):

    # If the device running the program has a GPU, no trimming will occur unless 
    # the user specifies so, and vice-versa
    def trim_gpu (self):
        if self.is_gpu_available() == True and self.trim == True:
            return self.random_trim()
        elif self.is_gpu_available() == True and self.trim == False:
            return self.data_generator()
        else:
            return self.random_trim()

print(len(pd.read_csv("./tools/data/structured_data/housing.csv")))

dataReader = DataReader("./tools/data/structured_data/housing.csv", trim=False, trim_ratio=0.5)
data = dataReader.data_generator()

print(len(data))

############################

# data_reader = DataReader('./data/housing.csv', trim=True, trim_ratio=0.5)

# print("Is GPU Available:", data_reader.is_gpu_available())
# print("Available GPUs:", data_reader.get_available_gpus())

# print("Before Trimming:", data_reader.data_generator().shape)
# print("After Trimming:", data_reader.trim_gpu().shape)