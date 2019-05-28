from keras.layers import Dropout
from typing import Tuple, Dict
from keras.layers.merge import concatenate
from ..layers import RectDense
from ..cnn import structure as cnn_structure
from ..mlp import structure as mlp_structure
from extra_keras_utils import set_seed

def structure(mlp:Dict, cnn:Dict, dense:Dict, dropout:Dict):
    set_seed(42)
    (input_cnn,), output_cnn = cnn_structure(**cnn)
    (input_mlp,), output_mlp = mlp_structure(**mlp)
    hidden = RectDense(**dense)(concatenate([output_cnn, output_mlp]))
    hidden = Dropout(**dropout)(hidden)
    return (input_cnn, input_mlp), hidden