import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model,
    Artifact,
)

@component(base_image="tensorflow/tensorflow:2.15.0", packages_to_install=[ "pandas==2.2.3", "scikit-learn==1.6.1"])
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    scaler: Input[Model],
    hyperparameters: dict,
    trained_model: Output[Model],
    training_dependencies: Output[Artifact],
):
    """
    Trains a dense tensorflow model.
    """

    def save_pip_freeze(filename="frozen_requirements.txt"):
        with open(filename, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, text=True)

    training_dependencies.path+=".txt"
    save_pip_freeze(training_dependencies.path)
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation, Concatenate
    from keras.layers import Input as KerasInput
    from tensorflow.keras.models import Model as KerasModel
    import pickle
    import pandas as pd
    import sklearn
    
    with open(train_data.path, 'rb') as pickle_file:
        X_train, y_train = pd.read_pickle(pickle_file)
    with open(val_data.path, 'rb') as pickle_file:
        X_val, y_val = pd.read_pickle(pickle_file)
    with open(scaler.path, 'rb') as pickle_file:
        scaler_ = pd.read_pickle(pickle_file)

    inputs = [KerasInput(shape=(1,), name=name) for name in X_train.columns]
    concatenated_inputs = Concatenate(name="input")(inputs)
    x = Dense(32, activation='relu', name="dense_0")(concatenated_inputs)
    x = Dense(64, name="dense_1")(x)
    x = Activation('relu')(x)
    x = Dense(128, name="dense_2")(x)
    x = Activation('relu')(x)
    x = Dense(256, name="dense_3")(x)
    x = Activation('relu')(x)
    output = Dense(y_train.shape[1], activation='sigmoid', name="dense_4")(x)
    model = KerasModel(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    model.summary()

    train_inputs = {name: X_train[[name]].to_numpy() for name in X_train.columns}
    val_inputs = {name: X_val[[name]].to_numpy() for name in X_val.columns}
    
    epochs = hyperparameters["epochs"]
    history = model.fit(
        train_inputs,
        y_train,
        epochs=epochs,
        validation_data=(val_inputs, y_val),
        verbose=True
    )
    print("Training of model is complete")
    
    trained_model.path += ".keras"
    model.save(trained_model.path)
    
    
@component(base_image="tensorflow/tensorflow:2.15.0", packages_to_install=["tf2onnx==1.16.1", "onnx==1.17.0", "pandas==2.2.3", "scikit-learn==1.6.1"])
def convert_keras_to_onnx(
    keras_model: Input[Model],
    onnx_model: Output[Model],
):
    import tf2onnx, onnx
    import keras
    import tensorflow as tf
    
    trained_keras_model = keras.saving.load_model(keras_model.path)
    input_signature = [tf.TensorSpec(input_.shape, input_.dtype, input_.name) for input_ in trained_keras_model.inputs]
    trained_keras_model.output_names = ['output']
    onnx_model_proto, _ = tf2onnx.convert.from_keras(trained_keras_model, input_signature)
    
    onnx_model.path += ".onnx"
    onnx.save(onnx_model_proto, onnx_model.path)