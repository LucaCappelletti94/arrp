import pandas as pd
import numpy as np
from typing import Dict, Callable
from .utils import load_settings, balance, get_cell_lines
from .paths import get_input_model_selection_path, get_input_model_validation_path, get_output_model_validation_path, get_epigenomic_data_path, get_classes_path, get_nucleotides_sequences_path, get_classes_path, get_train_path, get_test_path
from .paths import get_input_model_validation_path, get_output_model_selection_path
from .load_csv import load_nucleotides_sequences, load_classes, load_epigenomic_data

def selection_holdouts_generator(target:str, cell_line:str, task:Dict, balance_mode:str, outer:int):
    settings = load_settings(target)
    for inner in range(settings["selection_holdouts"]):
        epigenomic_data_train = load_epigenomic_data(get_epigenomic_data_path(get_train_path(get_input_model_selection_path(target, cell_line, outer, inner))))
        epigenomic_data_test = load_epigenomic_data(get_epigenomic_data_path(get_test_path(get_input_model_selection_path(target, cell_line, outer, inner))))
        nucleotides_sequences_train, _, _ = load_nucleotides_sequences(get_nucleotides_sequences_path(get_train_path(get_input_model_selection_path(target, cell_line, outer, inner))))
        nucleotides_sequences_test, _, _ = load_nucleotides_sequences(get_nucleotides_sequences_path(get_test_path(get_input_model_selection_path(target, cell_line, outer, inner))))
        classes_train = load_classes(get_classes_path(get_train_path(get_output_model_selection_path(target, cell_line, task["name"], outer, inner))))
        classes_test = load_classes(get_classes_path(get_test_path(get_output_model_selection_path(target, cell_line, task["name"], outer, inner))))
        yield balance(
            (epigenomic_data_train, epigenomic_data_test, nucleotides_sequences_train, nucleotides_sequences_test, classes_train, classes_test),
            balance_mode,
            "+".join(task["positive"]),
            "+".join(task["negative"]),
            settings["balance"]
        )

def validation_holdouts_generator(target:str, cell_line:str, task:Dict, balance_mode:str):
    settings = load_settings(target)
    for outer in range(settings["validation_holdouts"]):
        epigenomic_data_train = load_epigenomic_data(get_epigenomic_data_path(get_train_path(get_input_model_validation_path(target, cell_line, outer))))
        epigenomic_data_test = load_epigenomic_data(get_epigenomic_data_path(get_test_path(get_input_model_validation_path(target, cell_line, outer))))
        nucleotides_sequences_train, _, _ = load_nucleotides_sequences(get_nucleotides_sequences_path(get_train_path(get_input_model_validation_path(target, cell_line, outer))))
        nucleotides_sequences_test, _, _ = load_nucleotides_sequences(get_nucleotides_sequences_path(get_test_path(get_input_model_validation_path(target, cell_line, outer))))
        classes_train = load_classes(get_classes_path(get_train_path(get_output_model_validation_path(target, cell_line, task["name"], outer))))
        classes_test = load_classes(get_classes_path(get_test_path(get_output_model_validation_path(target, cell_line, task["name"], outer))))
        yield balance(
            (epigenomic_data_train, epigenomic_data_test, nucleotides_sequences_train, nucleotides_sequences_test, classes_train, classes_test),
            balance_mode,
            "+".join(task["positive"]),
            "+".join(task["negative"]),
            settings["balance"]
        )


def tasks_generator(target:str):
    settings = load_settings(target)
    for cell_line in get_cell_lines(target):
        for task in settings["tasks"]:
            for balance_mode, boolean in task["balancing"].items():
                if boolean:
                    yield (
                        target,
                        cell_line,
                        task,
                        balance_mode
                    )