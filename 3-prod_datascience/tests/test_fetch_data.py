from kfp import local
from kfp import dsl
import pandas as pd
from fetch_data import fetch_data
import pytest
import shutil
import os


def test_fetch_data():
    local.init(runner=local.SubprocessRunner())
    
    task = fetch_data()
    output_dataset = pd.read_csv(task.outputs["dataset"].path)
    expected_columns = ['spotify_id', 'name_x', 'artists_x', 'popularity', 'daily_rank',
           'daily_movement', 'weekly_movement', 'country', 'snapshot_date_x',
           'album_name', 'album_release_date', 'name_y', 'artists_y',
           'snapshot_date_y', 'is_explicit', 'duration_ms', 'danceability',
           'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
           'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    
    assert output_dataset.columns.tolist() == expected_columns

@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists("local_outputs"):
        shutil.rmtree("local_outputs")

if __name__ == "__main__":
    test_fetch_data()