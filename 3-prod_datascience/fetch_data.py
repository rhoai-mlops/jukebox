import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)

@component(base_image='python:3.9', packages_to_install=["feast==0.36.0", "psycopg2>=2.9", "dask[dataframe]", "s3fs", "pandas"])
def fetch_data(
    dataset: Output[Dataset]
):
    """
    Fetches data from URL
    """
    
    import pandas as pd
    import yaml    
    
    song_properties = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet')
    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')
    
    data = song_rankings.merge(song_properties, on='spotify_id', how='left')
    
    dataset.path += ".csv"
    data.metadata = {"song_properties": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet", "song_rankings": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet" }
    data.to_csv(dataset.path, index=False, header=True)

@component(base_image='python:3.9', packages_to_install=["feast==0.40.0", "psycopg2>=2.9", "dask[dataframe]", "s3fs", "pandas"])
def fetch_data_from_feast(
    dataset: Output[Dataset]
):
    """
    Fetches data from Feast
    """
    
    import feast
    import pandas as pd
    from datetime import datetime
    import yaml


    with open('../6-feature-store/feast-info/feature_store.yaml', 'r') as file:
        fs_config_yaml = yaml.safe_load(file)
    
    fs_config = feast.repo_config.RepoConfig(**fs_config_yaml)
    fs = feast.FeatureStore(config=fs_config)

    # Fetch the first X users latest values
    entity_df = pd.DataFrame.from_dict(
    {
        "transaction_id": list(range(10000)),
        "event_timestamp": [
            datetime.now()
            ]*10000
        }
    )

    features = [
        "transaction_features:distance_from_last_transaction",
        "transaction_features:ratio_to_median_purchase_price",
        "transaction_features:used_chip",
        "transaction_features:used_pin_number",
        "transaction_features:online_order",
        "transaction_features:fraud",
    ]

    dataset = fs.get_historical_features(entity_df=entity_df, features=features).to_df()
