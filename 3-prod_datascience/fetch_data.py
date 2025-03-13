import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)

@component(base_image='python:3.9', packages_to_install=["dask[dataframe]==2024.8.0", "s3fs==2025.2.0", "pandas==2.2.3"])
def fetch_data(
    dataset: Output[Dataset]
):
    """
    Fetches data from URL
    """
    
    import pandas as pd
    import yaml
    import os
    
    song_properties = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet')
    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')
    
    data = song_rankings.merge(song_properties, on='spotify_id', how='left')
    
    dataset.path += ".csv"
    dataset.metadata = {"song_properties": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet", "song_rankings": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet" }
    data.to_csv(dataset.path, index=False, header=True)


@component(base_image='python:3.9', packages_to_install=["dvc[s3]==3.1.0", "dask[dataframe]==2024.8.0", "s3fs==2025.2.0", "pandas==2.2.3"])
def fetch_data_from_dvc(
    dataset: Output[Dataset],
    cluster_domain: str,
    git_version: str,
):
    """
    Fetches data from DVC
    """
    
    import pandas as pd
    import yaml
    import dvc
    import configparser
    import os
    import subprocess
    import git
    
    def run_command(command, cwd=None, env=None):
        result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
        return result.stdout.strip()
        
    def read_hash(dvc_file_path):
        with open(dvc_file_path, 'r') as file:
            dvc_data = yaml.safe_load(file)
            md5_hash = dvc_data['outs'][0]['md5']
        return md5_hash

    git_username = os.environ.get('username')
    git_password = os.environ.get('password')
    current_path = os.environ.get("PATH", "")
    new_path = f"{current_path}:/.local/bin"
    os.environ["PATH"] = new_path
    
    print("Updated PATH:", os.environ["PATH"])

    namespace = os.environ.get("namespace").split('-')[0]
    os.chdir("/tmp")

    run_command(f"git clone https://{git_username}:{git_password}@gitea-gitea.{cluster_domain}/{namespace}/jukebox.git")
    os.chdir("/tmp/jukebox")
    try:
        run_command(f"git checkout {git_version}")
    except Exception as e:
        print(e)
        print(f"Could not check out version {git_version}")
    run_command("dvc pull | rc=$?")

    config = configparser.ConfigParser()
    config.read('.dvc/config')

    song_properties = pd.read_parquet("song_properties.parquet")
    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')
    
    data = song_rankings.merge(song_properties, on='spotify_id', how='left')
    
    dataset.path += ".csv"
    dvc_hash = read_hash("song_properties.parquet.dvc")
    dataset.metadata = {"DVC training data hash": dvc_hash} | {section: str(dict(config.items(section))) for section in config.sections()}
    data.to_csv(dataset.path, index=False, header=True)


@component(base_image='python:3.9', packages_to_install=["feast==0.40.0", "psycopg2>=2.9", "dask-expr==1.1.10", "s3fs==2024.6.1", "psycopg_pool==3.2.3", "psycopg==3.2.3", "pandas==2.2.3", "numpy==1.26.4"])
def fetch_data_from_feast(
    version: str,
    dataset: Output[Dataset]
):
    """
    Fetches data from Feast
    """
    
    import feast
    import pandas as pd
    import numpy as np

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

    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')
    # Feast will remove rows with identical id and date so we add a small delta to each
    microsecond_deltas = np.arange(0, len(song_rankings))*2
    song_rankings['snapshot_date'] = pd.to_datetime(song_rankings['snapshot_date'])
    song_rankings['snapshot_date'] = song_rankings['snapshot_date'] + pd.to_timedelta(microsecond_deltas, unit='us')

    feature_service = fs.get_feature_service("serving_fs")

    data = fs.get_historical_features(entity_df=song_rankings, features=feature_service).to_df()

    features = [f.name for f in feature_service.feature_view_projections[0].features]
    
    dataset.metadata = {"song_properties": "serving_fs", "song_rankings": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet", "features": features}
    dataset.path += ".csv"
    data.to_csv(dataset.path, index=False, header=True)