# kfp imports
import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)
from kfp import kubernetes

# Misc imports
import os

# Component imports
from fetch_data import fetch_data, fetch_data_from_feast
from data_validation import validate_data
from data_preprocessing import preprocess_data
from train_model import train_model, convert_keras_to_onnx
from evaluate_model import evaluate_keras_model_performance, validate_onnx_model
from save_model import push_to_model_registry

######### Pipeline definition #########

data_connection_secret_name = 'aws-connection-models'

# Create pipeline
@dsl.pipeline(
  name='training-pipeline',
  description='We train an amazing model 🚂'
)
def training_pipeline(hyperparameters: dict, model_name: str, version: str, cluster_domain: str, model_storage_pvc: str, prod_flag: bool):
    # Fetch Data
    fetch_task = fetch_data()
    # kubernetes.use_secret_as_env(
    #     fetch_task,
    #     secret_name='aws-connection-data',
    #     secret_key_to_env={
    #         'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
    #         'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
    #         'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
    #         'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
    #     },
    # )

    # Validate Data
    data_validation_task = validate_data(dataset = fetch_task.outputs["dataset"])

    # Pre-process Data
    pre_processing_task = preprocess_data(in_data = fetch_task.outputs["dataset"])
    pre_processing_task.after(data_validation_task)

    # Train Keras model
    training_task = train_model(
        train_data = pre_processing_task.outputs["train_data"], 
        val_data = pre_processing_task.outputs["val_data"],
        scaler = pre_processing_task.outputs["scaler"],
        hyperparameters = hyperparameters,
    )

    # Convert Keras model to ONNX
    convert_task = convert_keras_to_onnx(keras_model = training_task.outputs["trained_model"])

    # Evaluate Keras model performance
    model_evaluation_task = evaluate_keras_model_performance(
        model = training_task.outputs["trained_model"],
        test_data = pre_processing_task.outputs["test_data"],
        scaler = pre_processing_task.outputs["scaler"],
        label_encoder = pre_processing_task.outputs["label_encoder"],
        previous_model_metrics = {"accuracy":0.1},
    )

    # Validate that the Keras -> ONNX conversion was successful
    model_validation_task = validate_onnx_model(
        keras_model = training_task.outputs["trained_model"],
        onnx_model = convert_task.outputs["onnx_model"],
        test_data = pre_processing_task.outputs["test_data"],
        scaler = pre_processing_task.outputs["scaler"],
    )

    # Register model to the Model Registry
    register_model_task = push_to_model_registry(
        model_name = model_name, 
        version = version,
        cluster_domain = cluster_domain,
        prod_flag = prod_flag,
        model = convert_task.outputs["onnx_model"],
        metrics = model_evaluation_task.outputs["metrics"],
        dataset = fetch_task.outputs["dataset"],
        scaler = pre_processing_task.outputs["scaler"],
        label_encoder = pre_processing_task.outputs["label_encoder"],
    )
    kubernetes.use_secret_as_env(
        register_model_task,
        secret_name=data_connection_secret_name,
        secret_key_to_env={
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
        },
    )
    # Set PVC to store model artifacts
    register_model_task.after(model_validation_task)
    kubernetes.mount_pvc(
        register_model_task,
        pvc_name=model_storage_pvc,
        mount_path='/models',
    )
    

if __name__ == '__main__':
    metadata = {
        "hyperparameters": {
            "epochs": 2
        },
        "model_name": "jukebox",
        "version": "0.0.2",
        "cluster_domain": "apps.cluster-vx2f5.vx2f5.sandbox784.opentlc.com",
        "model_storage_pvc": "jukebox-model-pvc",
        "prod_flag": False
    }
        
    namespace_file_path =\
        '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint =\
        f'https://ds-pipeline-dspa.{namespace}.svc:8443'

    sa_token_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    with open(sa_token_file_path, 'r') as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert =\
        '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'

    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert
    )

    client.create_run_from_pipeline_func(
        training_pipeline,
        arguments=metadata,
        experiment_name="training",
        enable_caching=True
    )