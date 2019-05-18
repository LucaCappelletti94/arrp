from tqdm import tqdm as cli_tqdm, tqdm_notebook
from .is_notebook import is_notebook

tqdm = tqdm_notebook if is_notebook() else cli_tqdm