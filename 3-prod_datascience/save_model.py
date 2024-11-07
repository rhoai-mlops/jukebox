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

@component(
    base_image='python:3.9',
    packages_to_install=[
        'pip==24.2',  
        'setuptools>=65.0.0', 
        'model-registry==0.2.9'
    ]
)

def push_to_model_registry(
    model_name: str,
    version: str, 
    cluster_domain: str,
    model: Input[Model],
    metrics: Input[Metrics],
    scaler: Input[Model],
    label_encoder: Input[Model],
    dataset: Input[Dataset]
):
    from os import environ, path, makedirs
    from datetime import datetime
    from model_registry import ModelRegistry
    import shutil
    import json

    # Save to PVC
    makedirs("/models/artifacts", exist_ok=True)
    shutil.copyfile(model.path, f"/models/{model_name}.onnx")
    shutil.copyfile(scaler.path, f"/models/artifacts/scaler.pkl")
    shutil.copyfile(label_encoder.path, f"/models/artifacts/label_encoder.pkl")

    # Save to Model Registry
    model_object_prefix = model_name if model_name else "model"
    version = version if version else datetime.now().strftime('%y%m%d%H%M')
    cluster_domain = cluster_domain

    namespace_file_path =\
        '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read()
        
    def _register_model(author_name , cluster_domain, model_object_prefix, version):
        registry = ModelRegistry(server_address=f"https://" + author_name + "-registry-service." + cluster_domain, port=443, author=author_name, is_secure=False)
        registered_model_name = model_object_prefix
        version_name = version
        metadata = {
            "accuracy": str(metrics.metadata['Accuracy']),
            "dataset": json.dumps(dataset.metadata)
        }
        
        rm = registry.register_model(
            registered_model_name,
            "to-be-updated",
            model_format_name="onnx",
            model_format_version="1",
            version=version_name,
            description=f"Model {registered_model_name} version {version}",
            metadata=metadata
        )
        print("Model registered successfully")

    # Register the model
    _register_model(namespace, cluster_domain, model_object_prefix, version)