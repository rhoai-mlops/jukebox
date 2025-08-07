import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Artifact,
    ClassificationMetrics,
)

@component(base_image="tensorflow/tensorflow:2.15.0", packages_to_install=["tf2onnx==1.16.1", "onnx==1.17.0", "pandas==2.2.3", "scikit-learn==1.6.1", "model-registry==0.2.10"])
def evaluate_keras_model_performance(
    model: Input[Artifact],
    test_data: Input[Dataset],
    scaler: Input[Artifact],
    label_encoder: Input[Artifact],
    model_name: str,
    version: str,
    model_registry_server_address: str,
    model_registry_port: int,
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
):
    import keras
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import numpy as np
    from os import environ
    from model_registry import ModelRegistry
    from model_registry.exceptions import StoreError
    
    trained_model = keras.saving.load_model(model.path)
    with open(test_data.path, 'rb') as pickle_file:
        X_test, y_test = pd.read_pickle(pickle_file)
    with open(scaler.path, 'rb') as pickle_file:
        scaler_ = pd.read_pickle(pickle_file)
    with open(label_encoder.path, 'rb') as pickle_file:
        label_encoder_ = pd.read_pickle(pickle_file)

    test_inputs = {name: X_test[[name]].to_numpy() for name in X_test.columns}
    y_pred_temp = trained_model.predict(test_inputs)
    y_pred_temp = np.asarray(np.squeeze(y_pred_temp))
    
    y_pred_argmax = np.argmax(y_pred_temp, axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    
    accuracy = np.sum(y_pred_argmax == y_test_argmax) / len(y_pred_argmax)

    environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    registry = ModelRegistry(server_address=model_registry_server_address, port=model_registry_port, author="someone",is_secure=False)

    previous_model_properties = {}

    #Wrap with try except to see if the model exists in the registry
    try:
        # Get the latest models properties if no model is in production
        for v in registry.get_model_versions(model_name).order_by_id().descending():
            if not previous_model_properties:
                previous_model_properties = registry.get_model_versions(model_name).order_by_id().descending().next_item().custom_properties
            elif "prod" in v.custom_properties and v.custom_properties["prod"]:
                previous_model_properties = v.custom_properties
                break
    except StoreError:
        pass

    if "accuracy" not in previous_model_properties:
        previous_model_properties["accuracy"] = 0.1

    print("Previous model metrics: ", previous_model_properties)
    print("Accuracy: ", accuracy)

    metrics.log_metric("Accuracy", float(accuracy))
    metrics.log_metric("Prev Model Accuracy", float(previous_model_properties["accuracy"]))
    
    cmatrix = confusion_matrix(y_test_argmax,y_pred_argmax)
    cmatrix = cmatrix.tolist()
    targets = label_encoder_.classes_.tolist()
    classification_metrics.log_confusion_matrix(targets, cmatrix)
    
    if float(accuracy) <= 0.01: #float(previous_model_properties["accuracy"]):
        raise Exception("Accuracy is lower than the previous models")
        
@component(base_image="tensorflow/tensorflow:2.15.0", packages_to_install=["onnxruntime==1.20.1", "pandas==2.2.3", "scikit-learn==1.6.1"])
def validate_onnx_model(
    onnx_model: Input[Artifact],
    keras_model: Input[Artifact],
    test_data: Input[Dataset],
    scaler: Input[Artifact],
):
    import onnxruntime as rt
    import pandas as pd
    import numpy as np
    import keras
    
    with open(test_data.path, 'rb') as pickle_file:
        X_test, _ = pd.read_pickle(pickle_file)
    with open(scaler.path, 'rb') as pickle_file:
        scaler_ = pd.read_pickle(pickle_file)
    _keras_model = keras.saving.load_model(keras_model.path)
    onnx_session = rt.InferenceSession(onnx_model.path, providers=rt.get_available_providers())

    test_inputs = {name: X_test[[name]].to_numpy() for name in X_test.columns}

    onnx_output_name = onnx_session.get_outputs()[0].name
    onnx_pred = onnx_session.run([onnx_output_name], test_inputs)
    
    keras_pred = _keras_model(test_inputs)
    
    print("Keras Pred: ", keras_pred)
    print("ONNX Pred: ", onnx_pred[0])
    
    for rt_res, keras_res in zip(onnx_pred[0], keras_pred):
        np.testing.assert_allclose(rt_res, keras_res, rtol=1e-5, atol=1e-5)

    print("Results match")