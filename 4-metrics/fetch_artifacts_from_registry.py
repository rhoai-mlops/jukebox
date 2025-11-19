import os
from os import environ

import boto3
import urllib3
from model_registry import ModelRegistry

urllib3.disable_warnings()


def get_s3_client(s3_endpoint):
    session = boto3.session.Session()
    s3_access_key = environ.get('AWS_ACCESS_KEY_ID')
    s3_secret_key = environ.get('AWS_SECRET_ACCESS_KEY')

    return session.client(
        service_name='s3',
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint,
        verify=False,
    )


def fetch_artifacts_from_registry(token, artifacts, pipeline_namespace,
                                  model_registry_url, model_name, model_version,
                                  author_name, bucket_name="pipeline",
                                  debug=False):
    # Set up the model registry connection
    registry = ModelRegistry(server_address=model_registry_url, port=443,
                             user_token=token, author=author_name, is_secure=False)

    # Get model metadata
    version = registry.get_model_version(model_name, model_version)

    # See what pipeline run created this model
    try:
        pipeline_run_id = version.custom_properties['pipeline_run_id']
    except Exception:
        raise Exception(
            f"Model version {model_version} has no 'pipeline_run_id' property."
        )

    if debug: print("**** reference object prefix is " +
                    f"kfp-training-pipeline/{pipeline_run_id}")

    # Get the S3 client
    # s3_endpoint = f"http://minio-service.{pipeline_namespace}.svc.cluster.local:9000"
    s3_endpoint = f"https://minio-api-{pipeline_namespace}.apps.ocp.the511.local"
    s3_client = get_s3_client(s3_endpoint)

    # This gives us the timestamp of the latest object from
    # the reference pipeline run.
    try:
        refs = s3_client.list_objects_v2(
            Bucket="pipeline",
            Prefix=f"kfp-training-pipeline/{pipeline_run_id}",
            )['Contents']
    except Exception:
        raise Exception("No artifacts found for " +
                        f"kfp-training-pipeline/{pipeline_run_id} " +
                        "- was it executed by an OpenShift pipeline?")

    sorted(refs, key=lambda obj: obj['LastModified'])
    ref_ts = refs[-1]['LastModified']

    if debug: print(f"**** reference timestamp for artifacts is {ref_ts}")

    # Fetch all objects from kfp-training-pipeline prefix and
    # throw away anything newer than ref_ts
    objs = [obj for obj in s3_client.list_objects_v2(
                                Bucket="pipeline",
                                Prefix="kfp-training-pipeline",
                                MaxKeys=100000,
                            )['Contents'] if obj['LastModified'] <= ref_ts]

    if debug: print(f"**** got {len(objs)} objects of age >= {ref_ts}")

    # Only retain the latest version of any object
    # that matches the artifact name
    latest_found = {}
    for obj in objs:
        pcomps = obj['Key'].split('/')
        shortname = pcomps[-3] + '/' + pcomps[-1]

        if shortname not in artifacts:
            if debug: print(f"Skipping irrelevant {shortname}")
            continue
        if shortname not in latest_found.keys():
            if debug: print(f"Adding first instance of {shortname}")
            latest_found[shortname] = {
                'key': obj['Key'],
                'ts': obj['LastModified']
            }
            continue
        if obj['LastModified'] < latest_found[shortname]['ts']:
            if debug: print(f"Skipping {shortname} from " +
                            f"{obj['LastModified']} - it is older than " +
                            f"{latest_found[shortname]['ts']}")
            continue

        if debug: print(f"Replacing {shortname} from " +
                        f"{latest_found[shortname]['ts']} " +
                        f"with one from {obj['LastModified']}")
        latest_found[shortname] = {
            'key': obj['Key'],
            'ts': obj['LastModified']
        }

    # Download artifacts and return their locations
    save_paths = {}
    for artifact in artifacts:
        if artifact in latest_found.keys():
            # Create local directory structure to match the artifact path
            local_path = artifact
            local_dir = os.path.dirname(local_path)

            # Create directory if it doesn't exist
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            try:
                s3_client.download_file(bucket_name,
                                        latest_found[artifact]['key'],
                                        local_path)
                save_paths[artifact] = os.path.abspath(local_path)
                print(f"Successfully downloaded {artifact} to {local_path} " +
                      f"from {latest_found[artifact]['key']}")
            except Exception as e:
                print(f"Error downloading {artifact} " +
                      f"from {latest_found[artifact]['key']}: {e}")
        else:
            print(f"Could not find artifact in object store: {artifact}")

    return save_paths
