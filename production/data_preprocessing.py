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

@component(packages_to_install=["pandas", "scikit-learn"])
def preprocess_transactiondb_data(
    in_data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    scaler: Output[Model],
) -> NamedTuple('outputs', class_weights=dict):
    """
    Takes the dataset and preprocesses it to better train on the fraud detection model.
    The preprocessing consists of:
    1. Splitting the dataset into training, validation, and testing.
    2. Creating a scaler which scales down the training dataset. This scaler is saved as an artifact.
    3. Calculates the class weights, which will later be used during the training.
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight
    import pandas as pd
    import pickle
    import numpy as np
    from typing import NamedTuple
    
    df = pd.read_csv(in_data.path)
    print(df.head())
    X = df.drop(columns = ['transaction_id','repeat_retailer','distance_from_home', 'fraud', 'event_timestamp'])
    y = df['fraud']

    # Split the data into training and testing sets so you have something to test the trained model with.

    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle = False)

    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, stratify = y_train)

    # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier for the model to learn than random (and potentially large) values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.

    st_scaler = StandardScaler()

    X_train = st_scaler.fit_transform(X_train.values)
    
    train_data.path += ".pkl"
    val_data.path += ".pkl"
    test_data.path += ".pkl"
    scaler.path += ".pkl"
    
    with open(train_data.path, "wb") as handle:
        pickle.dump((X_train, y_train), handle)
    with open(val_data.path, "wb") as handle:
        pickle.dump((X_val, y_val), handle)
    with open(test_data.path, "wb") as handle:
        pickle.dump((X_test, y_test), handle)
    with open(scaler.path, "wb") as handle:
        pickle.dump(st_scaler, handle)

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.

    class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
    class_weights = {i : class_weights[i] for i in range(len(class_weights))}
    
    outputs = NamedTuple('outputs', class_weights=dict)
    return outputs(class_weights)