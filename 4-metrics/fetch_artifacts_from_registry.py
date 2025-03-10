from model_registry import ModelRegistry
import os
import boto3
from os import environ

def download_file_from_s3(s3_endpoint, bucket_name, object_key, download_path):
    session = boto3.session.Session()

    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')

    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint,
    )

    try:
        s3_client.download_file(bucket_name, object_key, download_path)
        print(f"File downloaded successfully to {download_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def fetch_artifacts_from_registry(artifacts, pipeline_namespace, model_registry_url, model_name, model_version, author_name, bucket_name="pipeline"):
    # Set up the model registry connection
    registry = ModelRegistry(server_address=model_registry_url, port=443, author=author_name, is_secure=False)
    version = registry.get_model_version(model_name, model_version)
    pipeline_run_id = version.custom_properties['pipeline_run_id'] #NOTE: This assumes that caching is disabled as we download the artifacts from S3 assuming that they are placed in the pipeline version folder.

    save_paths = {}
    for artifact in artifacts:
        download_file_from_s3(f"http://minio-service.{pipeline_namespace}.svc.cluster.local:9000", bucket_name, f"kfp-training-pipeline/{pipeline_run_id}/{artifact}", artifact.split("/")[-1])
        save_paths[artifact] = os.path.abspath(artifact.split("/")[-1])

    return save_paths