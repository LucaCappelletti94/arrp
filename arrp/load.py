import pandas as pd
import numpy as np
from .tasks import get_task_path, get_cell_line_path, get_model_selection_path, get_model_validation_path, get_holdout_path, get_balancing_path, get_default_file_names, is_nucleotide_sequence_file

def get_full_model_validation_path(target:str, cell_line:str, task:str, balance_mode:str):
    return get_balancing_path(get_model_validation_path(get_task_path(get_cell_line_path(target, cell_line), task)), balance_mode)

def get_full_model_selection_path(target:str, cell_line:str, task:str, balance_mode:str, holdout:int):
    return get_balancing_path(get_holdout_path(get_model_selection_path(get_task_path(get_cell_line_path(target, cell_line), task)), balance_mode), holdout)

def get_full_path(target:str, cell_line:str, task:str, balance_mode:str, holdout=None):
    if holdout is None:
        return get_full_model_validation_path(target, cell_line, task, balance_mode)
    return get_full_model_selection_path(target, cell_line, task, balance_mode, holdout)

def load_csv(path:str, name:str):
    df = pd.read_csv(
        "{path}/{name}.csv".format(
            path=path,
            name=name
        ), 
        index_col=0
    )
    if is_nucleotide_sequence_file(name):
        return df.values.reshape(-1, 200, 5)
    return df

def load(target:str, cell_line:str, task:str, balance_mode:str, holdout:int=None):
    path = get_full_path(target, cell_line, task, balance_mode, holdout)
    return [
        load_csv(path, name) for name in get_default_file_names()
    ]
