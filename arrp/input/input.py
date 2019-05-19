from typing import Dict
from ..utils import get_cell_lines, tqdm
from ..load_csv import load_raw_epigenomic_data, load_raw_nucleotides_sequences
from ..store_csv import store_epigenomic_data, store_nucleotides_sequences
from ..paths import get_input_model_selection_path, get_input_model_validation_path, get_test_path, get_train_path, get_epigenomic_data_path, get_nucleotides_sequences_path, get_holdout_path
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
import numpy as np

def get_epigenomic_data_train_path(path:str)->str:
    return get_epigenomic_data_path(get_train_path(path))

def get_epigenomic_data_test_path(path:str)->str:
    return get_epigenomic_data_path(get_test_path(path))

def get_nucleotides_sequences_train_path(path:str)->str:
    return get_nucleotides_sequences_path(get_train_path(path))

def get_nucleotides_sequences_test_path(path:str)->str:
    return get_nucleotides_sequences_path(get_test_path(path))

def is_cached(path:str)->bool:
    return all([
        os.path.exists(sub_path) for sub_path in (
            get_epigenomic_data_train_path(path),
            get_epigenomic_data_test_path(path),
            get_nucleotides_sequences_train_path(path),
            get_nucleotides_sequences_test_path(path)
        )
    ])

def job(epigenomic_data:pd.DataFrame, nucleotides_sequences:np.ndarray, nucleotides_sequences_index:np.ndarray, nucleotides_sequences_columns:np.ndarray, random_state:int, test_size:float, path:str):
    epigenomic_data_train, epigenomic_data_test, nucleotides_sequences_train, nucleotides_sequences_test, nucleotides_sequences_index_train, nucleotides_sequences_index_test = train_test_split(
        epigenomic_data, nucleotides_sequences, nucleotides_sequences_index, random_state=random_state, test_size=test_size
    )
    store_epigenomic_data(get_epigenomic_data_train_path(path), epigenomic_data_train)
    store_epigenomic_data(get_epigenomic_data_path(get_test_path(path)), epigenomic_data_test)
    store_nucleotides_sequences(get_nucleotides_sequences_train_path(path), nucleotides_sequences_train, nucleotides_sequences_index_train, nucleotides_sequences_columns)
    store_nucleotides_sequences(get_nucleotides_sequences_test_path(path), nucleotides_sequences_test, nucleotides_sequences_index_test, nucleotides_sequences_columns)

def kwarged_job(kwargs):
    job(**kwargs)

def build_input(target:str, settings:Dict):
    for cell_line in tqdm(get_cell_lines(target), desc="Cell lines input"):
        epigenomic_data = load_raw_epigenomic_data(target, cell_line)
        nucleotides_sequences, nucleotides_sequences_index, nucleotides_sequences_columns = load_raw_nucleotides_sequences(target, cell_line)
        validation_path = get_input_model_validation_path(target, cell_line)
        if not is_cached(validation_path):
            kwarged_job({
                "epigenomic_data":epigenomic_data,
                "nucleotides_sequences":nucleotides_sequences,
                "nucleotides_sequences_index":nucleotides_sequences_index,
                "nucleotides_sequences_columns":nucleotides_sequences_columns,
                "random_state":settings["test_random_state"],
                "test_size":settings["test_size"],
                "path":validation_path
            })
        epigenomic_data_train, _, nucleotides_sequences_train, _, nucleotides_sequences_index_train, _ = train_test_split(
            epigenomic_data, nucleotides_sequences, nucleotides_sequences_index, random_state=settings["test_random_state"], test_size=settings["test_size"]
        )
        jobs = [
            {
                "epigenomic_data":epigenomic_data_train,
                "nucleotides_sequences":nucleotides_sequences_train,
                "nucleotides_sequences_index":nucleotides_sequences_index_train,
                "nucleotides_sequences_columns":nucleotides_sequences_columns,
                "random_state":settings["validation_starting_random_state"]+holdout,
                "test_size":settings["validation_size"],
                "path":get_input_model_selection_path(target, cell_line, holdout)
            } for holdout in range(settings["holdouts"]) if not is_cached(get_input_model_selection_path(target, cell_line, holdout))
        ]
        if len(jobs):
            with Pool(cpu_count()) as p:
                list(tqdm(p.imap(kwarged_job, jobs), desc="Holdouts", leave=False, total=len(jobs)))