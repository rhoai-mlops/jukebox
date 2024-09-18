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

@component(base_image="tensorflow/tensorflow", packages_to_install=[ "pandas", "scikit-learn"])
def train_fraud_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    scaler: Input[Model],
    class_weights: dict,
    hyperparameters: dict,
    trained_model: Output[Model]
):
    """
    Trains a dense tensorflow model.
    """
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation
    import pickle
    import pandas as pd
    import sklearn
    
    with open(train_data.path, 'rb') as pickle_file:
        X_train, y_train = pd.read_pickle(pickle_file)
    with open(val_data.path, 'rb') as pickle_file:
        X_val, y_val = pd.read_pickle(pickle_file)
    with open(scaler.path, 'rb') as pickle_file:
        st_scaler = pd.read_pickle(pickle_file)
        
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    
    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim = X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    epochs = hyperparameters["epochs"]
    history = model.fit(X_train, y_train, epochs=epochs, \
                        validation_data=(st_scaler.transform(X_val.values),y_val), \
                        verbose = True, class_weight = class_weights)
    print("Training of model is complete")
    
    trained_model.path += ".keras"
    model.save(trained_model.path)
    
    
@component(base_image="tensorflow/tensorflow", packages_to_install=["tf2onnx", "onnx", "pandas", "scikit-learn"])
def convert_keras_to_onnx(
    keras_model: Input[Model],
    onnx_model: Output[Model],
):
    import tf2onnx, onnx
    import keras
    import tensorflow as tf
    
    trained_keras_model = keras.saving.load_model(keras_model.path)
    input_signature = [tf.TensorSpec(trained_keras_model.inputs[0].shape, trained_keras_model.inputs[0].dtype, name='input')]
    trained_keras_model.output_names = ['output']
    onnx_model_proto, _ = tf2onnx.convert.from_keras(trained_keras_model, input_signature)
    
    onnx_model.path += ".onnx"
    onnx.save(onnx_model_proto, onnx_model.path)