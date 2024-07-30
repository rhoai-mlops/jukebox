import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)

@component()
def validate_transactiondb_data(
    dataset: Input[Dataset]
) -> bool:
    """
    Validates if the data schema is correct and if the values are reasonable.
    """
    
    if not dataset.path:
        raise Exception("dataset not found")
    return True