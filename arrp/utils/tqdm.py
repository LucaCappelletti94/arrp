from tqdm import tqdm as tqdm_cli, tqdm_notebook
from .is_notebook import is_notebook

tqdm = tqdm_notebook if is_notebook() else tqdm_cli