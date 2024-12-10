import kfp
from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)
from kfp import kubernetes

@component(packages_to_install=["pyarrow", "pandas"])
def extract_data(
    data: Output[Dataset],
):
    import pandas as pd

    song_properties = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet')
    data.path += ".parquet"
    song_properties.to_parquet(data.path, index=False)

@component(packages_to_install=["pandas", "pyarrow"])
def transform_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
):
    import pandas as pd
    
    df = pd.read_parquet(input_data.path)
    df.columns = map(str.lower, df.columns)
    
    output_data.path += ".parquet"
    df.to_parquet(output_data.path, index=False)

@component(packages_to_install=["boto3", "pandas", "botocore"])
def load_data(
    data: Input[Dataset]
):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')


    s3_client = boto3.client(
        's3', aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    
    s3_client.upload_file(data.path, bucket_name, "song_properties.parquet")
    
@dsl.pipeline(
  name='ETL Pipeline',
  description='Moves and transforms data from transactions data storage (postgresql) to S3.'
)
def etl_pipeline():
    extract_task = extract_data()

    transform_task = transform_data(input_data=extract_task.outputs["data"])

    load_task = load_data(data=transform_task.outputs["output_data"])
    kubernetes.use_secret_as_env(
        load_task,
        secret_name='aws-connection-data',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
        },
    )
    

def main():
    COMPILE=False
    if COMPILE:
        kfp.compiler.Compiler().compile(etl_pipeline, 'song-properties-etl.yaml')
    else:
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
            etl_pipeline,
            experiment_name="song-properties-etl",
            namespace=namespace,
            enable_caching=True
        )

if __name__ == '__main__':
    main()