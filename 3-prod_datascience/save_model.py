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
        'boto3',
        'model-registry==0.2.9'
    ]
)

def push_to_model_registry(
    model_name: str,
    version: str, 
    model: Input[Model],
    metrics: Input[Metrics],
    scaler: Input[Model],
    label_encoder: Input[Model],
):
    from os import environ, path, makedirs
    from datetime import datetime
    from boto3 import client
    from model_registry import ModelRegistry
    import shutil

    # Save to PVC
    makedirs("/models/artifacts", exist_ok=True)
    shutil.copyfile(model.path, f"/models/{model_name}.onnx")
    shutil.copyfile(scaler.path, f"/models/artifacts/scaler.pkl")
    shutil.copyfile(label_encoder.path, f"/models/artifacts/label_encoder.pkl")

    # Save to S3
    model_object_prefix = model_name if model_name else "model"
    s3_endpoint_url = environ.get('AWS_S3_ENDPOINT')
    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')
    s3_bucket_name = environ.get('AWS_S3_BUCKET')
    author_name = environ.get('AUTHOR_NAME', 'default_author') 
    version = version if version else datetime.now().strftime('%y%m%d%H%M')

    def _initialize_s3_client(s3_endpoint_url, s3_access_key, s3_secret_key):
        print('Initializing S3 client')
        s3_client = client(
            's3', aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            endpoint_url=s3_endpoint_url,
        )
        return s3_client

    # Initialize the S3 client
    s3_client = _initialize_s3_client(
        s3_endpoint_url=s3_endpoint_url,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key
    )

    # use Git hash instead?
    def _generate_artifact_name(artifact_file_name, version=''):
        artifact_name, artifact_extension = path.splitext(path.basename(artifact_file_name))
        artifact_version_file_name = f'{artifact_name}-{version}{artifact_extension}'
        print(artifact_version_file_name)
        return artifact_version_file_name


    def _do_upload(s3_client, model_path, object_name, s3_bucket_name):
        print(f'Uploading model to {object_name}')
        try:
            s3_client.upload_file(model_path, s3_bucket_name, object_name)
        except Exception as e:
            print(f'S3 upload to bucket {s3_bucket_name} at {s3_endpoint_url} failed: {e}')
            raise
        print(f'Model uploaded and available as "{object_name}"')

    # Upload the model - how to make the path better?
    model_artifact_s3_path = f"/models/{model_name}/1/{_generate_artifact_name(f'{model_name}.onnx', version)}"
    scaler_artifact_s3_path = f"/models/{model_name}/1/artifacts/{_generate_artifact_name(scaler.path, version)}"
    label_encoder_artifact_s3_path = f"/models/{model_name}/1/artifacts/{_generate_artifact_name(label_encoder.path, version)}"
    
    _do_upload(s3_client, model.path, model_artifact_s3_path, s3_bucket_name)
    _do_upload(s3_client, scaler.path, scaler_artifact_s3_path, s3_bucket_name)
    _do_upload(s3_client, label_encoder.path, label_encoder_artifact_s3_path, s3_bucket_name)

    def _register_model(author_name, model_object_prefix, version, s3_endpoint_url, model_name):
        namespace = environ.get("NAMESPACE")
        registry = ModelRegistry(server_address=f"http://model-registry-service.{namespace}.svc.cluster.local", port=8080, author=author_name, is_secure=False)
        registered_model_name = model_object_prefix
        version_name = version
        metadata = {
            # "metrics": metrics,
            "license": "apache-2.0",
            "scaler_artifact": f"s3://{s3_endpoint_url}{scaler_artifact_s3_path}",
            "label_encoder_artifact": f"s3://{s3_endpoint_url}{label_encoder_artifact_s3_path}",
        }
        print(metadata)
        
        rm = registry.register_model(
            registered_model_name,
            f"s3://{s3_endpoint_url}{model_artifact_s3_path}",
            model_format_name="onnx",
            model_format_version="1",
            version=version_name,
            description=f"Model {model_name} version {version}",
            metadata=metadata
        )
        print("Model registered successfully")

    # Register the model
    _register_model(author_name, model_object_prefix, version, s3_endpoint_url.split("http://")[-1], _generate_artifact_name(model.path, version))