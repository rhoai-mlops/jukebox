import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model,
)
from typing import NamedTuple

@component(base_image="tensorflow/tensorflow", packages_to_install=["pandas", "scikit-learn"])
def preprocess_data(
    in_data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    scaler: Output[Model],
    label_encoder: Output[Model],
):
    """
    Takes the dataset and preprocesses it to better train on the fraud detection model.
    The preprocessing consists of:
    1. Splitting the dataset into training, validation, and testing.
    2. Creating a scaler which scales down the training dataset. This scaler is saved as an artifact.
    3. Calculates the class weights, which will later be used during the training.
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
    from sklearn.utils import class_weight
    import pandas as pd
    import pickle
    import numpy as np
    from typing import NamedTuple
    import tensorflow as tf
    
    df = pd.read_csv(in_data.path)
    df = df.dropna()
    print(df.head())

    if "features" in in_data.metadata:
        X = df[in_data.metadata["features"]["list"]]
    else:
        X = df[['is_explicit', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    y = df['country']

    label_encoder_ = LabelEncoder()
    y_encoded = label_encoder_.fit_transform(y)
    y_one_hot = tf.keras.utils.to_categorical(y_encoded)

    # Split the data into training and testing sets so you have something to test the trained model with.
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.2, shuffle = False, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, stratify = y_train, random_state=42)

    # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes 
    # it a lot easier for the model to learn than random (and potentially large) values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about 
    # the global distribution of variables (which is influenced by the test set) into the training set.
    scaler_ = MinMaxScaler()
    scaled_x_train = pd.DataFrame(scaler_.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    scaled_x_val = pd.DataFrame(scaler_.transform(X_val), index=X_val.index, columns=X_val.columns)
    scaled_x_test = pd.DataFrame(scaler_.transform(X_test), index=X_test.index, columns=X_test.columns).astype(np.float32)
    
    train_data.path += ".pkl"
    val_data.path += ".pkl"
    test_data.path += ".pkl"
    scaler.path += ".pkl"
    label_encoder.path += ".pkl"
    
    with open(train_data.path, "wb") as handle:
        pickle.dump((scaled_x_train, y_train), handle)
    with open(val_data.path, "wb") as handle:
        pickle.dump((scaled_x_val, y_val), handle)
    with open(test_data.path, "wb") as handle:
        pickle.dump((scaled_x_test, y_test), handle)
    with open(scaler.path, "wb") as handle:
        pickle.dump(scaler_, handle)
    with open(label_encoder.path, "wb") as handle:
        pickle.dump(label_encoder_, handle)