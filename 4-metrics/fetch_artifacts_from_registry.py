from model_registry import ModelRegistry
import os
import boto3
from os import environ


def get_s3_client(s3_endpoint):
    session = boto3.session.Session()
    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')

    return session.client(
        service_name='s3',
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint,
    )


def find_artifact_path(s3_client, bucket_name, pipeline_run_id, artifact):
    # Split artifact path to get component name
    component = artifact.split('/')[0]
    artifact_filename = artifact.split('/')[-1]

    # Look for folders under the kfp-training-pipeline directory
    pipeline_prefix = "kfp-training-pipeline/"
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=pipeline_prefix,
            Delimiter='/'
        )
        if 'CommonPrefixes' in response:
            # Check pipeline_run_id existence or fallback to last found
            found = False
            for obj in response['CommonPrefixes']:
                obj_id = obj.get('Prefix').split('/')[-2]
                found |= obj_id == pipeline_run_id
            if not found:
                print(f"Warning: pipeline_run_id {pipeline_run_id} not found! Falling back to {obj_id}")
                pipeline_run_id = obj_id
        else:
            print(f"Error: No folders found under {pipeline_prefix}")
            return None, None
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return None, None

    # Look for folders and artifact under each pipeline run ID for the component
    # This solution assumes that caching is enabled and artifacts are stored under a pipeline run ID folder but not always the last one
    for pipeline_run_id in response['CommonPrefixes']:
        component_prefix = f"kfp-training-pipeline/{pipeline_run_id.get('Prefix').split('/')[-2]}/{component}/"
        print(f"INFO: Looking for the artifact {artifact_filename} in the folder {component_prefix}")
        try:
            response_internal = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=component_prefix,
                Delimiter='/'
            )

            if 'CommonPrefixes' in response_internal:
                # Find the folder that contains our artifact
                for prefix_info in response_internal['CommonPrefixes']:
                    random_id_folder = prefix_info['Prefix']

                    # Check if this folder contains our artifact file
                    file_response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=random_id_folder
                    )

                    if 'Contents' in file_response:
                        for obj in file_response['Contents']:
                            if obj['Key'].endswith(artifact_filename):
                                return random_id_folder, obj['Key']
            else:
                print(f"WARNING: No folders found under {component_prefix}")
        except Exception as e:
            print(f"Error listing S3 objects: {e}")
            return None, None
    return None, None

def download_file_from_s3(s3_endpoint, bucket_name, object_key, download_path):
    s3_client = get_s3_client(s3_endpoint)

    try:
        s3_client.download_file(bucket_name, object_key, download_path)
        print(f"File downloaded successfully to {download_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")


def fetch_artifacts_from_registry(token, artifacts, pipeline_namespace, model_registry_url, model_name, model_version, author_name, bucket_name="pipeline"):
    # Set up the model registry connection
    registry = ModelRegistry(server_address=model_registry_url, port=443, user_token=token, author=author_name, is_secure=False)
    version = registry.get_model_version(model_name, model_version)
    pipeline_run_id = version.custom_properties['pipeline_run_id'] #NOTE: This assumes that caching is disabled as we download the artifacts from S3 assuming that they are placed in the pipeline version folder.

    s3_endpoint = f"http://minio-service.{pipeline_namespace}.svc.cluster.local:9000"
    s3_client = get_s3_client(s3_endpoint)

    save_paths = {}
    for artifact in artifacts:
        # Find the actual path with random ID
        artifact_folder_path, artifact_file_key = find_artifact_path(s3_client, bucket_name, pipeline_run_id, artifact)

        if artifact_folder_path and artifact_file_key:
            # Create local directory structure to match the artifact path
            local_path = artifact
            local_dir = os.path.dirname(local_path)

            # Create directory if it doesn't exist
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            try:
                download_file_from_s3(s3_endpoint, bucket_name, artifact_file_key, local_path)
                save_paths[artifact] = os.path.abspath(local_path)
                print(f"Successfully downloaded {artifact} to {local_path}")
            except Exception as e:
                print(f"Error downloading {artifact}: {e}")
        else:
            print(f"Could not find artifact folder for {artifact}")

    return save_paths
