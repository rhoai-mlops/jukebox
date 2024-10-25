from feast import FeatureService

from features import song_properties

song_properties_fs = FeatureService(
    name="serving_fs",
    features=[
        song_properties[[
            "is_explicit",
            "duration_ms",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo"
            ]]
    ]
)