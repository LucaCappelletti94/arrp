import pandas as pd
import numpy as np
from typing import Dict, Callable
from .utils import load_settings, balance, get_cell_lines, load_raw_nucleotides_sequences, load_raw_classes, load_raw_epigenomic_data
from holdouts_generator import random_holdouts, holdouts_generator

def balanced_generator(generator, mode:str, pos:str, neg:str, settings:Dict)->Callable:
    if generator is None:
        return None
    def wrapper():
        for (training, testing), sub_generator in generator():
            balanced = balance(training, testing, mode, pos, neg, settings)
            training, testing = balanced[:len(training)], balanced[len(training):]
            yield (training, testing), balanced_generator(sub_generator, mode, pos, neg, settings)
    return wrapper

def balanced_holdouts_generator(target:str, cell_line:str, task:Dict, balance_mode:str):
    settings = load_settings(target)
    epigenomic_data = load_raw_epigenomic_data(target, cell_line)
    nucleotides_sequences, _, _ = load_raw_nucleotides_sequences(target, cell_line)
    classes = pd.DataFrame(load_raw_classes(target, cell_line)[task["positive"]].any(axis=1), columns=["+".join(task["positive"])])
    generator = holdouts_generator(
        epigenomic_data, nucleotides_sequences, classes,
        holdouts=random_holdouts(**settings["holdouts"]),
        cache=True,
        cache_dir=".holdouts/{target}/{cell_line}/{name}".format(
            target=target,
            cell_line=cell_line,
            name=task["name"].replace(" ", "_")
        )
    )
    return balanced_generator(generator, balance_mode, "+".join(task["positive"]), "+".join(task["negative"]), settings["balance"])

def tasks_generator(target:str):
    settings = load_settings(target)
    for cell_line in get_cell_lines(target):
        for task in settings["tasks"]:
            for balance_mode, boolean in task["balancing"].items():
                if boolean:
                    yield (target, cell_line, task, balance_mode)