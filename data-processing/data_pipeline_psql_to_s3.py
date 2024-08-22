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

@component(packages_to_install=["psycopg2", "pandas"])
def extract_data(
    source_con_details: dict,
    data: Output[Dataset],
):
    import psycopg2
    import pandas as pd
    import os

    conn = psycopg2.connect(
        host=source_con_details['host'],
        database=os.environ["database_name"],
        user=os.environ["database_user"],
        password=os.environ["database_password"],
    )
    query = f"SELECT * FROM {source_con_details['table']}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    data.path += ".csv"
    df.to_csv(data.path, index=False)

@component(packages_to_install=["pandas", "pyarrow"])
def transform_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
):
    import pandas as pd
    
    df = pd.read_csv(input_data.path)
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

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key)
    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    bucket.upload_file(data.path, "card_transaction_data.parquet")

@dsl.pipeline(
  name='ETL Pipeline',
  description='Moves and transforms data from transactions data storage (postgresql) to S3.'
)
def etl_pipeline(source_con_details: dict):
    extract_task = extract_data(source_con_details=source_con_details)
    kubernetes.use_secret_as_env(
        extract_task,
        secret_name="transactionsdb-info",
        secret_key_to_env={
            'database-name': 'database_name',
            'database-user': 'database_user',
            'database-password': 'database_password',
        }
    )

    transform_task = transform_data(input_data=extract_task.outputs["data"])

    load_task = load_data(data=transform_task.outputs["output_data"])
    kubernetes.use_secret_as_env(
        load_task,
        secret_name='aws-connection-feast-offline-store',
        secret_key_to_env={
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
        },
    )
    

def main():
    COMPILE=False
    if COMPILE:
        kfp.compiler.Compiler().compile(etl_pipeline, 'transactiondb-feast-etl.yaml')
    else:
        metadata = {
            "source_con_details": {
                "host": "transactionsdb.mlops-transactionsdb.svc.cluster.local",
                "table": "transactions.transactions",
            },
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
            etl_pipeline,
            arguments=metadata,
            experiment_name="transactiondb-feast-etl",
            namespace="mlops-feature-store",
            enable_caching=True
        )

if __name__ == '__main__':
    main()