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

USER = ""
DATASET = ""
CLUSTER_DOMAIN = ""

@component(packages_to_install=["pyarrow", "pandas"])
def extract_parquet_from_url(
    dataset: str,
    data: Output[Dataset],
):
    import pandas as pd

    parquet_data = pd.read_parquet(dataset)

    data.path += ".parquet"
    parquet_data.to_parquet(data.path, index=False)

@component(base_image='python:3.9', packages_to_install=["pyarrow==18.0.0", "pandas==2.2.3"])
def extract_parquet_from_s3(
    data: Output[Dataset],
):
    import pandas as pd
    import pyarrow.fs
    import os

    fs = pyarrow.fs.S3FileSystem(
            endpoint_override=os.environ.get('AWS_S3_ENDPOINT'),
            access_key=os.environ.get('AWS_ACCESS_KEY_ID'),
            secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )

    with fs.open_input_file('data/song_properties.parquet') as file:
        parquet_data = pd.read_parquet(file)

    data.path += ".parquet"
    parquet_data.to_parquet(data.path, index=False)

@component(packages_to_install=["pyarrow", "pandas"])
def add_date(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
):
    import pandas as pd
    from datetime import datetime

    df = pd.read_parquet(input_data.path)
    df.columns = map(str.lower, df.columns)
    df["snapshot_date"] = datetime.now()

    print(df)

    output_data.path += ".parquet"
    df.to_parquet(output_data.path, index=False)

@component(packages_to_install=["pyarrow", "pandas"])
def concat_datasets(
    input_data_1: Input[Dataset],
    input_data_2: Input[Dataset],
    output_data: Output[Dataset],
):
    import pandas as pd

    df1 = pd.read_parquet(input_data_1.path)
    df2 = pd.read_parquet(input_data_2.path)

    joined_df = pd.concat([df1, df2], ignore_index=True)
    joined_df = joined_df.drop_duplicates(subset=["spotify_id"], keep="last")

    output_data.path += ".parquet"
    joined_df.to_parquet(output_data.path, index=False)

@component(packages_to_install=["boto3", "pandas", "botocore"])
def push_to_s3(
    data: Input[Dataset]
):
    import os
    import boto3

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    bucket_name = os.environ.get('AWS_S3_BUCKET')


    s3_client = boto3.client(
        's3', aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )

    s3_client.upload_file(data.path, bucket_name, "song_properties.parquet")


@component(base_image='python:3.9', packages_to_install=["feast==0.40.0", "psycopg2>=2.9", "dask-expr==1.1.10", "s3fs==2024.6.1", "psycopg_pool==3.2.3", "psycopg==3.2.3", "pandas", "numpy"])
def materialize_changes(
):
    import feast
    import pandas as pd
    import numpy as np
    from datetime import datetime

    fs_config_json = {
        'project': 'music',
        'provider': 'local',
        'registry': {
            'registry_type': 'sql',
            'path': 'postgresql://feast:feast@feast:5432/feast',
            'cache_ttl_seconds': 60,
            'sqlalchemy_config_kwargs': {
                'echo': False, 
                'pool_pre_ping': True
            }
        },
        'online_store': {
            'type': 'postgres',
            'host': 'feast',
            'port': 5432,
            'database': 'feast',
            'db_schema': 'feast',
            'user': 'feast',
            'password': 'feast'
        },
        'offline_store': {'type': 'file'},
        'entity_key_serialization_version': 2
    }

    fs_config = feast.repo_config.RepoConfig(**fs_config_json)
    fs = feast.FeatureStore(config=fs_config)

    fs.materialize_incremental(datetime.now())


@component(packages_to_install=["dvc[s3]==3.1.0"])
def setup_dvc_repository_with_env_credentials(
    repo_url: str,
    dvc_data_url: str,
    email: str,
):
    import os
    import subprocess
    import git
    import yaml

    # Retrieve S3 info environment variables
    aws_endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    # Retrieve Git credentials from environment variables
    git_username = os.environ.get('username')
    git_password = os.environ.get('password')

    current_path = os.environ.get("PATH", "")
    new_path = f"{current_path}:/.local/bin"
    os.environ["PATH"] = new_path

    print("Updated PATH:", os.environ["PATH"])

    def run_command(command, cwd=None, env=None):
        result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
        return result.stdout.strip()


    if not git_username or not git_password:
        raise ValueError(f"Git credentials not found in environment variables: {git_username}, {git_password}")

    os.chdir("/tmp")
    
    # Construct the repository URL with credentials
    repo_url_with_credentials = repo_url.replace(
        "https://", f"https://{git_username}:{git_password}@"
    )
    repo_dir = "jukebox"

    # Clone the repository if not already cloned
    if not os.path.exists(f"/tmp/{repo_dir}"):
        print(f"Cloning repository from {repo_url_with_credentials}...")
        git.Repo.clone_from(repo_url_with_credentials, repo_dir)
    
    os.chdir(f"/tmp/{repo_dir}")
    print(os.listdir())

    def push_git(commit_message = "", to_commit = []):
        # Configure Git and commit changes
        git_repo = git.Repo(".")
        git_repo.config_writer().set_value("user", "email", email).release()
        git_repo.config_writer().set_value("user", "name", git_username).release()
        for file in to_commit:
            run_command(f"git add {file}")
        git_repo.index.commit(commit_message)
        
        # Push changes to the remote repository
        print("Pushing changes...")
        run_command("git push origin main")
        print("Push complete.")

    def read_hash(dvc_file_path):
        with open(dvc_file_path, 'r') as file:
            dvc_data = yaml.safe_load(file)
            md5_hash = dvc_data['outs'][0]['md5']
        return md5_hash

    init_flag = False
    try:
        run_command("dvc status")
    except Exception as e:
        init_flag = True
        print("An error occurred:", e)
    
    if init_flag:
        # Initialize and configure DVC
        print("Initializing DVC...")
        run_command("dvc init")
        run_command(f"dvc remote add --default s3-version {dvc_data_url}")
        run_command(f"dvc remote modify s3-version endpointurl {aws_endpoint_url}")
        run_command("dvc remote add data-source s3://data")
        run_command(f"dvc remote modify data-source endpointurl {aws_endpoint_url}")
        run_command(f"dvc import-url remote://data-source/song_properties.parquet --to-remote")
        import configparser
        config = configparser.ConfigParser()
        config.read('.dvc/config')
        print({section: dict(config[section]) for section in config.sections()})
        push_git(commit_message = "Initial data tracked", to_commit=["song_properties.parquet.dvc", ".dvc/config"])
    else:
        # Update DVC based on new data in S3
        print("Updating data version...")
        run_command(f"dvc update song_properties.parquet.dvc --to-remote")
        push_git(commit_message = f"Updated data to version {read_hash('song_properties.parquet.dvc')}", to_commit=["song_properties.parquet.dvc"])

@dsl.pipeline(
  name='ETL Pipeline',
  description='Moves and transforms data from transactions data storage (postgresql) to S3.'
)
def etl_pipeline(url_dataset: str, repo_url: str):
    # Extract our current data
    extract_s3_data_task = extract_parquet_from_s3()
    kubernetes.use_secret_as_env(
        extract_s3_data_task,
        secret_name='aws-connection-data',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
        },
    )

    # Extract the new data we want to add
    extract_url_data_task = extract_parquet_from_url(dataset=url_dataset)

    # Add the current time to our data
    add_date_task = add_date(input_data=extract_url_data_task.outputs["data"])

    concat_task = concat_datasets(
        input_data_1=extract_s3_data_task.outputs["data"],
        input_data_2=add_date_task.outputs["output_data"]
    )

    load_task = push_to_s3(data=concat_task.outputs["output_data"])
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

    materialize_task = materialize_changes()
    kubernetes.use_secret_as_env(
        materialize_task,
        secret_name='aws-connection-data',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
        },
    )
    materialize_task.after(load_task)

    setup_dvc_task = setup_dvc_repository_with_env_credentials(
        repo_url=repo_url,
        dvc_data_url="s3://data-cache",
        email="mlops@wizard.com",
    )
    kubernetes.use_secret_as_env(
        setup_dvc_task,
        secret_name='aws-connection-data',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
        },
    )
    kubernetes.use_secret_as_env(
        setup_dvc_task,
        secret_name='git-auth',
        secret_key_to_env={
            'username': 'username',
            'password': 'password',
        },
    )
    setup_dvc_task.after(materialize_task)
    # setup_dvc_task.set_caching_options(False)
    for task in [
        extract_s3_data_task,
        extract_url_data_task,
        add_date_task,
        concat_task,
        load_task,
        materialize_task,
        setup_dvc_task
    ]:
        task.set_caching_options(enable_caching=False)

def main():
    COMPILE=True
    if COMPILE:
        kfp.compiler.Compiler().compile(etl_pipeline, 'song-properties-etl.yaml')
    else:
        metadata = {
            "dataset_url": f"https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/{DATASET}",
            "repo_url": f"https://gitea-gitea.{CLUSTER_DOMAIN}/{USER}/jukebox.git",
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
            experiment_name="song-properties-etl",
            namespace=namespace,
            enable_caching=False
        )

if __name__ == '__main__':
    main()