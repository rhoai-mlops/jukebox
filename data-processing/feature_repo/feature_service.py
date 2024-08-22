from feast import FeatureService

from features import transactions_features

transactions_fs = FeatureService(
    name="training_fs",
    features=[
        transactions_features[["distance_from_last_transaction", "ratio_to_median_purchase_price", "used_chip", "used_pin_number", "online_order", "fraud"]]
    ]
)

transactions_fs = FeatureService(
    name="serving_fs",
    features=[
        transactions_features[["distance_from_last_transaction", "ratio_to_median_purchase_price", "used_chip", "used_pin_number", "online_order"]]
    ]
)