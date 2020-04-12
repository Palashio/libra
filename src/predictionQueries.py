import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from matplotlib import pyplot

from predictionModelCreation import getKerasModel

pd.set_option('display.max_columns', None)

def SingleRegressionQuery(dataset_path, user_def_label):
        data = pd.read_csv(dataset_path)
        data.fillna(0, inplace=True)
        
        categorical_columns = data.select_dtypes(exclude=["number"]).columns
        numeric_columns = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]

        if(len(categorical_columns) != 0):

            categorical_feature_mask = data.dtypes==object
            categorical_cols = data.columns[categorical_feature_mask].tolist()
            labeled_df = data[categorical_cols]

            enc = OneHotEncoder()
            enc.fit(labeled_df)
            onehotlabels = enc.transform(labeled_df).toarray()
            
            new_columns=list()
            for col, values in zip(labeled_df.columns, enc.categories_):
                new_columns.extend([col + '_' + str(value) for value in values])

            data = pd.concat([data, pd.DataFrame(onehotlabels, columns=new_columns)], axis='columns')

            for x in categorical_cols: del data[x]

        if(len(numeric_columns) != 0):
            scaler = StandardScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        y = data[user_def_label]
        del data[user_def_label]

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

        models=[]
        losses = []
        epochs = 5

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        i = 0
        model = getKerasModel(data, i)

        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
        models.append(history)
        losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])


        while(all(x > y for x, y in zip(losses, losses[1:]))):
             model = getKerasModel(data, i)
             history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])

             models.append(history)
             losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])
             print("The number of layers " + str(len(model.layers)))
             i += 1


        return models[i] 
        
 
SingleRegressionQuery("./data/housing.csv", "median_house_value")
