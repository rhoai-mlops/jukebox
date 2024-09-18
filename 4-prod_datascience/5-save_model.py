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
        'model-registry==0.2.3a1'
    ]
)

def push_to_model_registry(
    model_name: str,
    version: str, 
    model: Input[Model]
):
    from os import environ
    from datetime import datetime
    from boto3 import client
    from model_registry import ModelRegistry

    model_object_prefix = model_name if model_name else "model"
    s3_endpoint_url = environ.get('AWS_S3_ENDPOINT')
    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')
    s3_bucket_name = environ.get('AWS_S3_BUCKET')
    author_name = environ.get('AUTHOR_NAME', 'default_author') 
    version = version

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
    def _timestamp():
        return datetime.now().strftime('%y%m%d%H%M')

    def _generate_model_name(model_object_prefix, version=''):
        version = version if version else _timestamp()
        model_name = f'{model_object_prefix}-{version}.onnx'
        return model_name, version

    # Generate the model object name
    model_object_name, version = _generate_model_name(
        model_object_prefix, version=version
    )


    def _do_upload(s3_client, model_path, object_name, s3_bucket_name):
        print(f'Uploading model to {object_name}')
        try:
            s3_client.upload_file(model_path, s3_bucket_name, object_name)
        except Exception as e:
            print(f'S3 upload to bucket {s3_bucket_name} at {s3_endpoint_url} failed: {e}')
            raise
        print(f'Model uploaded and available as "{object_name}"')

    # Upload the model - how to make the path better?
    _do_upload(s3_client, model.path, "/models/fraud/1/" + model_object_name, s3_bucket_name)

    def _register_model(author_name, model_object_prefix, version, s3_endpoint_url, model_name):
        registry = ModelRegistry(server_address="http://model-registry-service.kubeflow.svc.cluster.local", port=8080, author=author_name, is_secure=False)
        registered_model_name = model_object_prefix
        version_name = version
        rm = registry.register_model(
            registered_model_name,
            f"s3://{s3_endpoint_url}/{model_name}",
            model_format_name="onnx",
            model_format_version="1",
            version=version_name,
            description=f"Example Model version {version}",
            metadata={
                # "accuracy": 3.14,
                "license": "apache-2.0",
                "stage": "test"
            }
        )
        print("Model registered successfully")

    # Register the model
    _register_model(author_name, model_object_prefix, version, s3_endpoint_url, model_object_name)