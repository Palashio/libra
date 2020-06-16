from data_reader import DataReader

data_reader = DataReader('./data/housing.csv')

print("Is GPU Available:", data_reader.is_gpu_available())
print("Available GPUs:", data_reader.get_available_gpus())