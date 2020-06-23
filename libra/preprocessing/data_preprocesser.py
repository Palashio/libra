import keras
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from os import listdir
from PIL import Image as PImage
from libra.data_generation.dataset_labelmatcher import get_similar_column
from libra.data_generation.grammartree import get_value_instruction
import cv2
from prince.ca import CA


def initial_preprocesser(data, instruction, preprocess, mca_threshold):
    # Scans for object columns just in case we have a datetime column that
    # isn't detected
    object_columns = [
        col for col,
                col_type in data.dtypes.iteritems() if col_type == 'object']

    # Handles dates without timestamps
    for col in object_columns:
        try:
            data[col] = pd.to_datetime(data[col], infer_datetime_format=True)
        except ValueError:
            pass

    # get target column
    target = get_similar_column(
        get_value_instruction(instruction), data)
    y = data[target]

    # remove rows where target is NaN
    data = data[y.notna()]
    y = y[y.notna()]

    del data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=49)

    data = {
        'train': pd.concat([X_train], axis=1),
        'test': pd.concat([X_test], axis=1)
    }
    # preprocess the dataset
    full_pipeline = None
    if preprocess:
        data, full_pipeline = structured_preprocesser(data, mca_threshold)
    else:
        data.fillna(0, inplace=True)

    y = {'train': y_train, 'test': y_test}

    return data, y, target, full_pipeline


# Preprocesses the data appropriately for single reg data
def structured_preprocesser(data, mca_threshold):
    # Preprocessing for datetime columns
    process_dates(data)

    # identifies the categorical and numerical columns
    categorical_columns = data['train'].select_dtypes(
        exclude=["number"]).columns
    numeric_columns = data['train'].columns[data['train'].dtypes.apply(
        lambda c: np.issubdtype(c, np.number))]

    # pipeline for numeric columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([], remainder="passthrough")

    if len(numeric_columns) != 0:
        full_pipeline.transformers.append(("num", num_pipeline, numeric_columns))

    if len(categorical_columns) != 0:
        combined = pd.concat([data['train'], data['test']], axis=0)

        mca_threshold = combined.shape[0]*.25 if mca_threshold is None else combined.shape[0] * mca_threshold

        if too_many_values(combined[categorical_columns], mca_threshold):
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="constant", fill_value="")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('transformer', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
                ('mca', CA(n_components=-1))
            ])
        else:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="constant", fill_value="")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

        full_pipeline.transformers.append(('cat', cat_pipeline, categorical_columns))

    train = full_pipeline.fit_transform(data['train'])

    train_cols = generate_column_labels(full_pipeline, numeric_columns)

    test = full_pipeline.transform(data['test'])

    test_cols = generate_column_labels(full_pipeline, numeric_columns)

    # Ternary clause because when running housing.csv,
    # the product of preprocessing is np array, but not when using landslide
    # data... not sure why
    data['train'] = pd.DataFrame(
        (train.toarray() if not isinstance(
            train,
            np.ndarray) else train),
        columns=train_cols)
    data['test'] = pd.DataFrame(
        (test.toarray() if not isinstance(
            test,
            np.ndarray) else test),
        columns=test_cols)

    return data, full_pipeline


def process_dates(data):
    for df in data.values():
        datetime_cols = df.select_dtypes('datetime64')

        for col in datetime_cols:
            df[f'{col}_DayOfWeek'] = df[col].dt.day_name()
            df[f'{col}_Year'] = df[col].dt.year
            df[f'{col}_Month'] = df[col].dt.month_name()
            df[f'{col}_MonthDay'] = df[col].dt.day

            del df[col]


# Sees if one hot encoding occurred, if not just uses numeric cols
def generate_column_labels(pipeline, numeric_cols):
    # Check if one hot encoding was performed
    if 'cat' in pipeline.named_transformers_:
        # If mca was used
        if isinstance(pipeline.named_transformers_['cat'][-1], CA):
            ca = pipeline.named_transformers_['cat'][-1]
            encoded_cols = [f'CA_{x}' for x in range(len(ca.eigenvalues_))]
            cols = [*list(numeric_cols), *encoded_cols]

        else:
            try:
                encoded_cols = pipeline.named_transformers_[
                    'cat']['one_hot_encoder'].get_feature_names()
                cols = [*list(numeric_cols), *encoded_cols]

            except Exception as error:
                # For debugging only
                print(error)
                cols = list(numeric_cols)

        return cols
    else:
        return numeric_cols


def clustering_preprocessor(data):
    # identifies the categorical and numerical columns
    categorical_columns = data.select_dtypes(exclude=["number"]).columns
    numeric_columns = data.columns[data.dtypes.apply(
        lambda c: np.issubdtype(c, np.number))]

    # pipeline for numeric columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # pipeline for categorical columns
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value="")),
        ('one_hot_encoder', OneHotEncoder()),
    ])

    # combine the two pipelines
    if len(numeric_columns) != 0 and len(categorical_columns) != 0:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_columns),
            ("cat", cat_pipeline, categorical_columns),
        ])
    elif len(numeric_columns) == 0:
        full_pipeline = ColumnTransformer([
            ("cat", cat_pipeline, categorical_columns),
        ])
    else:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_columns),
        ])

    data = full_pipeline.fit_transform(data)

    new_columns = generate_column_labels(full_pipeline, numeric_columns)

    return pd.DataFrame(data, columns=new_columns), full_pipeline


# Method to calculate how many columns the data set will
# have after one hot encoding
# Decides whether MCA is needed or not essentially
# mca_threshold is the len of the dataset * .25 to calculate the proportion of
# when to apply MCA
def too_many_values(data, mca_threshold):
    total_unique = 0
    for col in data:

        if total_unique > mca_threshold: return True
        # Use value_counts() due to same columns having strings and floats
        total_unique += len(data[col].value_counts())

    return False
