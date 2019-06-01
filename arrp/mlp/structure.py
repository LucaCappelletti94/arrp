from keras.layers import Dropout, Input
from ..layers import RectDense
from typing import Tuple, Dict, List
from extra_keras_utils import set_seed

def structure(dense:List[Dict], input_shape:Tuple[int]):
    set_seed(42)
    hidden = input_layer = Input(shape=input_shape, name="mlp")
    for d in dense:
        hidden = RectDense(**d["dense"])(hidden)
        hidden = Dropout(**d["dropout"])(hidden)
    return (input_layer,), hidden