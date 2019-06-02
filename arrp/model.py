import extra_keras_metrics
from keras.layers import Layer, Dense
from keras.models import Model
from typing import Tuple, List


def head(layer: Layer)->Layer:
    return Dense(1, activation="sigmoid")(layer)


def model(inputs: Tuple[Layer, Layer], output: Layer, metrics:List=None)->Model:
    model = Model(
        inputs=inputs,
        outputs=[head(output)]
    )
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["auprc", "auroc", "accuracy"] if metrics is None else metrics
    )
    return model
