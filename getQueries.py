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


def SingleRegressionQuery(dataset_path, user_request):
        data = pd.read_csv(dataset_path)
        
        #Performing One Hot Encoding and Standard Scaler
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

        if(len(numeric_columns) != 0):
            scaler = StandardScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        label = getLabelForPrediction(data)
        del data["median_house_value"]
        
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=49)


        
        model = getKerasModel()
        history = model.fit(X_train, y_train, epochs=30)

def getKerasModel(): 
    model = tf.keras.Sequential()
    model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def getLabelForPrediction(dataset):
    return dataset["median_house_value"]

    
SingleRegressionQuery("housing.csv", "print second value")
