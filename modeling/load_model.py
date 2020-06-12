#Loading the user defined model given that the user provides the model path.
#The user must provide the model Directory.
from keras.models import model_from_json
import os

def load_model(self,load_path):
    load_file=load_path[load_path.rindex("/")+1:load_path.rindex(".")]
    #loading model json file
    json_file = open(load_path, 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    # loading weights in the model
    loaded_model.load_weights(load_file+".h5")
    print("Model loaded...")
    return loaded_model
