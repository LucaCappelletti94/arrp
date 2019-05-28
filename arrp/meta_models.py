from .mlp import space as mlp_space
from .mlp import structure as mlp_structure
from .cnn import space as cnn_space
from .cnn import structure as cnn_structure
from .mmnn import space as mmnn_space
from .mmnn import structure as mmnn_structure
from .model_selection import model_selection
from gaussian_process import Space
from typing import Callable
from notipy_me import Notipy

def notipy_wrapper(target:str, name:str, structure:Callable, space:Space, notipy:bool):
    if notipy:
        with Notipy("bayesian selection for {name}".format(name=name), send_start_mail=True):
            model_selection(target, name, structure, space)
    else:
        model_selection(target, name, structure, space)
        
def mlp(target:str, notipy:bool=False):
    notipy_wrapper(target, "mlp", mlp_structure, mlp_space, notipy)

def cnn(target:str, notipy:bool=False):
    notipy_wrapper(target, "cnn", cnn_structure, cnn_space, notipy)

def mmnn(target:str, notipy:bool=False):
    notipy_wrapper(target, "mmnn", mmnn_structure, mmnn_space, notipy)