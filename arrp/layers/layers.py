from keras.layers import Conv1D, BatchNormalization, Layer, Dense, MaxPool1D
from typing import Callable, Dict

def NormConv1d(**Conv1D_kwargs):
    def wrapper(previous:Layer):
        return BatchNormalization()(
            Conv1D(
                filters=Conv1D_kwargs["filters"],
                kernel_size=(Conv1D_kwargs["kernel_size"],),
                padding="same",
                activation=Conv1D_kwargs["activation"],
            )(previous)
        )
    return wrapper

def RectLayer(layers:int, layer:Callable, **kwargs):
    def wrapper(previous:Layer):
        for _ in range(layers):
            previous = layer(**kwargs)(previous)
        return previous
    return wrapper

def PoolRectNormConv1d(layers:int, MaxPooling1D_kwargs:Dict, Conv1D_kwargs:Dict):
    def wrapper(previous:Layer):
        return MaxPool1D(**MaxPooling1D_kwargs)(RectLayer(layers, NormConv1d, **Conv1D_kwargs)(previous))
    return wrapper

def RectDense(layers:int, **kwargs):
    return RectLayer(layers, Dense, **kwargs)