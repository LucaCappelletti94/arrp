from .mlp import space as mlp_space
from .mlp import structure as mlp_structure
from .cnn import space as cnn_space
from .cnn import structure as cnn_structure
from .mmnn import space as mmnn_space
from .mmnn import structure as mmnn_structure
from .model_selection import model_selection
from gaussian_process import Space
from typing import Callable

def mlp(target:str):
    model_selection(target, "mlp", mlp_structure, mlp_space)

def cnn(target:str):
    model_selection(target, "cnn", cnn_structure, cnn_space)

def mmnn(target:str):
    model_selection(target, "mmnn", mmnn_structure, mmnn_space)