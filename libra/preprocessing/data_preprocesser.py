import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from libra.data_generation.dataset_labelmatcher import get_similar_column
from libra.data_generation.grammartree import get_value_instruction
from prince.ca import CA

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller


def initial_preprocesser(
        data,
        instruction,
        preprocess,
        ca_threshold,
        text,
        test_size=0.2,
        random_state=49):
    # Scans for object columns just in case we have a datetime column that
    # isn't detected

    if test_size < 0 or test_size > 1:
        raise Exception(
            'Test size cannot be {}, it should be a proportion between 0 and 1'.format(test_size))

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
        data, y, test_size=test_size, random_state=random_state)

    data = {
        'train': pd.concat([X_train], axis=1),
        'test': pd.concat([X_test], axis=1)
    }
    # preprocess the dataset
    full_pipeline = None
    if preprocess:
        data, full_pipeline = structured_preprocesser(data, ca_threshold, text)
    else:
        data.fillna(0, inplace=True)

    y = {'train': y_train, 'test': y_test}

    return data, y, target, full_pipeline


def structured_preprocesser(data, ca_threshold, text):

    # Preprocessing for datetime columns
    process_dates(data)

    # This will be used inside process_text once complete
    if len(text) > 0:
        text_preprocessing(data, text)

    # identifies the categorical and numerical columns
    categorical_columns = data['train'].select_dtypes(
        exclude=["number"]).columns
    numeric_columns = data['train'].columns[data['train'].dtypes.apply(
        lambda c: np.issubdtype(c, np.number))]

    # Removes text columns from categorical columns to use in separate pipeline
    categorical_columns = [
        cat_cols for cat_cols in categorical_columns if cat_cols not in text]

    full_pipeline = ColumnTransformer([], remainder="passthrough")

    if len(numeric_columns) != 0:
        # pipeline for numeric columns
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])

        full_pipeline.transformers.append(
            ("num", num_pipeline, numeric_columns))

    if len(text) != 0:
        # Each text col needs a separate pipeline
        for x in range(len(text)):
            full_pipeline.transformers.append(
                (f"text_{x}",
                 Pipeline(
                     [
                         ('test',
                          FunctionTransformer(
                              lambda x: np.reshape(
                                  x.to_numpy(),
                                  (-1,
                                   1)))),
                         ('imputer',
                          SimpleImputer(
                              strategy="constant",
                              fill_value="")),
                         ('raveler',
                          FunctionTransformer(
                              lambda x: x.ravel(),
                              accept_sparse=True)),
                         ('vect',
                          TfidfVectorizer()),
                         ('densifier',
                          FunctionTransformer(
                              lambda x: x.todense(),
                              accept_sparse=True)),
                         ('embedder',
                          FunctionTransformer(
                              text_embedder,
                              accept_sparse=True))]),
                    text[x]))

    if len(categorical_columns) != 0:
        combined = pd.concat([data['train'], data['test']], axis=0)

        ca_threshold = combined.shape[0] * \
            .25 if ca_threshold is None else combined.shape[0] * ca_threshold

        if too_many_values(combined[categorical_columns], ca_threshold):
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="constant", fill_value="")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('transformer', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
                ('ca', CA(n_components=-1))
            ])
        else:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="constant", fill_value="")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

        full_pipeline.transformers.append(
            ('cat', cat_pipeline, categorical_columns))

    train = full_pipeline.fit_transform(data['train'])

    train_cols = generate_column_labels(full_pipeline, numeric_columns, text)

    test = full_pipeline.transform(data['test'])

    test_cols = generate_column_labels(full_pipeline, numeric_columns, text)

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

# Preprocesses text for word embedding


def text_preprocessing(data, text_cols):

    lemmatizer = WordNetLemmatizer()
    combined = pd.concat([data['train'], data['test']], axis=0)

    spell = Speller(fast=True)
    for col in text_cols:
        combined[col] = combined[col].apply(
            lambda x: x.lower() if isinstance(x, str) else x)

    stop_words = set(stopwords.words('english'))

    for col in text_cols:
        preprocessed_text = []
        for words in combined[col]:
            if words is not np.nan:
                words = word_tokenize(words)
                words = [word for word in words if word.isalpha()]
                words = [word for word in words if word not in stop_words]
                words = [spell(word) for word in words]
                words = [lemmatizer.lemmatize(word) for word in words]

                preprocessed_text.append(' '.join(words))

            else:
                preprocessed_text.append(np.nan)

        combined[col] = preprocessed_text
    data['train'] = combined.iloc[:len(data['train'])]
    data['test'] = combined.iloc[len(data['train']):]


def text_embedder(text):

    total = list()
    for i in text:
        total.append(np.sum(i))

    return np.reshape(total, (-1, 1))

# Sees if one hot encoding occurred, if not just uses numeric cols


def generate_column_labels(full_pipeline, numeric_cols, text_cols):
    # Check if one hot encoding was performed
    if 'cat' in full_pipeline.named_transformers_:
        # If ca was used
        if isinstance(full_pipeline.named_transformers_['cat'][-1], CA):
            ca = full_pipeline.named_transformers_['cat'][-1]
            encoded_cols = [f'CA_{x}' for x in range(len(ca.eigenvalues_))]
            cols = [*list(numeric_cols), *encoded_cols, *text_cols]

        else:
            try:
                encoded_cols = full_pipeline.named_transformers_[
                    'cat']['one_hot_encoder'].get_feature_names()
                cols = [*list(numeric_cols), *encoded_cols, *text_cols]

            except Exception as error:
                # For debugging only
                print(error)
                cols = list(numeric_cols)

        return cols
    else:
        return [*list(numeric_cols), *text_cols]


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

    new_columns = generate_column_labels(
        full_pipeline, numeric_columns, text_cols=[])

    return pd.DataFrame(data, columns=new_columns), full_pipeline


# Method to calculate how many columns the data set will
# have after one hot encoding
# Decides whether CA is needed or not essentially
# mca_threshold is the len of the dataset * .25 to calculate the proportion of
# when to apply CA
def too_many_values(data, ca_threshold):
    total_unique = 0
    for col in data:

        if total_unique > ca_threshold:
            return True
        # Use value_counts() due to same columns having strings and floats
        total_unique += len(data[col].value_counts())

    return False
