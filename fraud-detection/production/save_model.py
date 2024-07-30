import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model,
)

@component()
def push_to_model_registry(
    model: Input[Model],
):
    pass