from keras.layers import Dropout, Input, MaxPooling1D, Flatten, Layer
from typing import Dict, Tuple
from ..layers import RectDense, PoolRectNormConv1d
from extra_keras_utils import set_seed

def structure(convolutionals:Dict, dense:Dict)->Tuple[Tuple[Layer], Layer]:
    set_seed(42)
    hidden = input_layer =  Input(shape=(200,5), name="cnn")
    for convolutional in convolutionals:
        hidden = PoolRectNormConv1d(**convolutional)(hidden)
    hidden = Flatten()(hidden)
    for d in dense:
        hidden = RectDense(**d)(hidden)
        hidden = Dropout(0.1)(hidden)
    return (input_layer,), hidden 