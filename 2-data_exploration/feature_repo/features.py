from datetime import timedelta

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, PushSource, RequestSource
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, Bool
from feast.data_format import ParquetFormat

music = Entity(name="music", join_keys=["spotify_id"])

music_source = FileSource(
    name="music_s3",
    path="s3://data/song_properties.parquet",
    s3_endpoint_override="minio-service.mlops.svc.cluster.local",
    file_format=ParquetFormat(),
    timestamp_field="snapshot_date",
)

song_properties = FeatureView(
    name = "song_properties",
    entities = [music],
    schema = [
        Field(name="is_explicit", dtype=Bool),
        Field(name="duration_ms", dtype=Int64),
        Field(name="danceability", dtype=Float32),
        Field(name="energy", dtype=Float32),
        Field(name="key", dtype=Int64),
        Field(name="loudness", dtype=Float32),
        Field(name="mode", dtype=Int64),
        Field(name="speechiness", dtype=Float32),
        Field(name="acousticness", dtype=Float32),
        Field(name="instrumentalness", dtype=Float32),
        Field(name="liveness", dtype=Float32),
        Field(name="valence", dtype=Float32),
        Field(name="tempo", dtype=Float32),
    ],
    source = music_source,
)