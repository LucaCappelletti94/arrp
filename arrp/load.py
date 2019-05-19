import pandas as pd
import numpy as np
from typing import Dict
from .utils import load_settings, balance, get_cell_lines
from .paths import get_input_model_validation_path, get_output_model_validation_path, get_epigenomic_data_path, get_classes_path, get_nucleotides_sequences_path, get_train_path, get_test_path
from .paths import get_input_model_selection_path, get_output_model_selelection_path
from .load_csv import load_nucleotides_sequences, load_classes, load_epigenomic_data

def load_validation(target:str, cell_line:str, task:Dict, balance_mode:str):
    settings = load_settings(target)
    epigenomic_data_train = load_epigenomic_data(get_train_path(get_input_model_validation_path(target, cell_line)))
    epigenomic_data_test = load_epigenomic_data(get_test_path(get_input_model_validation_path(target, cell_line)))
    nucleotides_sequences_train, _, _ = load_nucleotides_sequences(get_train_path(get_input_model_validation_path(target, cell_line)))
    nucleotides_sequences_test, _, _ = load_nucleotides_sequences(get_test_path(get_input_model_validation_path(target, cell_line)))
    classes_train = load_classes(get_train_path(get_output_model_validation_path(target, cell_line, task["name"])))
    classes_test = load_classes(get_test_path(get_output_model_validation_path(target, cell_line, task["name"])))
    return balance(
        (epigenomic_data_train, epigenomic_data_test, nucleotides_sequences_train, nucleotides_sequences_test, classes_train, classes_test),
        balance_mode,
        task["positive"],
        task["negative"],
        settings["balance"]
    )


def holdouts_generator(target:str, cell_line:str, task:Dict, balance_mode:str):
    settings = load_settings(target)
    for holdout in range(settings["holdouts"]):
        epigenomic_data_train = load_epigenomic_data(get_train_path(get_input_model_selection_path(target, cell_line, holdout)))
        epigenomic_data_test = load_epigenomic_data(get_test_path(get_input_model_selection_path(target, cell_line, holdout)))
        nucleotides_sequences_train, _, _ = load_nucleotides_sequences(get_train_path(get_input_model_selection_path(target, cell_line, holdout)))
        nucleotides_sequences_test, _, _ = load_nucleotides_sequences(get_test_path(get_input_model_selection_path(target, cell_line, holdout)))
        classes_train = load_classes(get_train_path(get_output_model_selelection_path(target, cell_line, task["name"], holdout)))
        classes_test = load_classes(get_test_path(get_output_model_selelection_path(target, cell_line, task["name"], holdout)))
        yield balance(
            (epigenomic_data_train, epigenomic_data_test, nucleotides_sequences_train, nucleotides_sequences_test, classes_train, classes_test),
            balance_mode,
            task["positive"],
            task["negative"],
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