import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model,
    ClassificationMetrics,
)

@component(base_image="tensorflow/tensorflow", packages_to_install=["tf2onnx", "onnx", "pandas", "scikit-learn"])
def evaluate_keras_model_performance(
    model: Input[Model],
    test_data: Input[Dataset],
    scaler: Input[Model],
    label_encoder: Input[Model],
    previous_model_metrics: dict,
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
):
    import keras
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    trained_model = keras.saving.load_model(model.path)
    with open(test_data.path, 'rb') as pickle_file:
        X_test, y_test = pd.read_pickle(pickle_file)
    with open(scaler.path, 'rb') as pickle_file:
        scaler_ = pd.read_pickle(pickle_file)
    with open(label_encoder.path, 'rb') as pickle_file:
        label_encoder_ = pd.read_pickle(pickle_file)
    
    y_pred_temp = trained_model.predict(scaler_.transform(X_test.values))
    y_pred_temp = np.asarray(np.squeeze(y_pred_temp))
    
    y_pred_argmax = np.argmax(y_pred_temp, axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    
    accuracy = np.sum(y_pred_argmax == y_test_argmax) / len(y_pred_argmax)
    
    metrics.log_metric("Accuracy", accuracy)
    metrics.log_metric("Prev Model Accuracy", previous_model_metrics["accuracy"])
    
    cmatrix = confusion_matrix(y_test_argmax,y_pred_argmax)
    cmatrix = cmatrix.tolist()
    targets = label_encoder_.classes_.tolist()
    classification_metrics.log_confusion_matrix(targets, cmatrix)
    
    if accuracy <= previous_model_metrics["accuracy"]:
        raise Exception("Accuracy is lower than the previous models")
        
@component(base_image="tensorflow/tensorflow", packages_to_install=["onnxruntime", "pandas", "scikit-learn"])
def validate_onnx_model(
    onnx_model: Input[Model],
    keras_model: Input[Model],
    test_data: Input[Dataset],
    scaler: Input[Model],
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
    
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    onnx_pred = onnx_session.run([onnx_output_name], {onnx_input_name: scaler_.transform(X_test.values).astype(np.float32)})
    
    keras_pred = _keras_model(scaler_.transform(X_test.values).astype(np.float32))
    
    print("Keras Pred: ", keras_pred)
    print("ONNX Pred: ", onnx_pred[0])
    
    for rt_res, keras_res in zip(onnx_pred[0], keras_pred):
        np.testing.assert_allclose(rt_res, keras_res, rtol=1e-5, atol=1e-5)

    print("Results match")