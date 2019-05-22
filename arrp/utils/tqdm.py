from tqdm import tqdm as tqdm_cli, tqdm_notebook
from .is_notebook import is_notebook

_tqdm = tqdm_notebook if is_notebook() else tqdm_cli

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, dynamic_ncols=True)