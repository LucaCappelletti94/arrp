from gaussian_process import Space
from ..mlp import space as mlp_space
from ..cnn import space as cnn_space

space = Space({
    "mlp":mlp_space,
    "cnn":cnn_space,
    "dense":{
        "layers":[1,3],
        "units":[8, 64],
        "activation":"relu"
    },
    "dropout":{
        "rate":[0.0, 0.5]
    }
})