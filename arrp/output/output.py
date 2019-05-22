from typing import Dict
from ..utils import get_cell_lines, tqdm
from ..load_csv import load_raw_classes
from ..store_csv import store_classes
from ..paths import get_classes_path, get_train_path, get_test_path, get_output_model_validation_path, get_output_model_selection_path
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split

def get_classes_train_path(path:str)->str:
    return get_classes_path(get_train_path(path))

def get_classes_test_path(path:str)->str:
    return get_classes_path(get_test_path(path))

def is_cached(path:str)->bool:
    return all([
        os.path.exists(sub_path) for sub_path in (
            get_classes_train_path(path),
            get_classes_test_path(path)
        )
    ])

def job(classes:pd.DataFrame, random_state:int, test_size:float, path:str):
    if is_cached(path):
        return 0
    classes_train, classes_test = train_test_split(
        classes, random_state=random_state, test_size=test_size
    )
    store_classes(get_classes_train_path(path), classes_train)
    store_classes(get_classes_test_path(path), classes_test)

def kwarged_job(kwargs):
    job(**kwargs)


def build_output(target:str, settings:Dict):
    for cell_line in tqdm(get_cell_lines(target), desc="Cell lines output"):
        classes = load_raw_classes(target, cell_line)
        for task in tqdm(settings["tasks"], desc="Tasks output", leave=False):
            task_classes = pd.DataFrame(classes[task["positive"]].any(axis=1), columns=["+".join(task["positive"])])
            jobs = []
            for outer in range(settings["validation_holdouts"]):
                seed = settings["validation_starting_random_state"]+outer
                jobs.append({
                    "classes":task_classes,
                    "random_state":seed,
                    "test_size":settings["validation_test_size"],
                    "path":get_output_model_validation_path(target, cell_line, task["name"], outer)
                })
                classes_train, _ = train_test_split(
                    task_classes, random_state=seed, test_size=settings["validation_test_size"]
                )
                jobs += [
                    {
                        "classes":classes_train,
                        "random_state":settings["selection_starting_random_state"]+inner,
                        "test_size":settings["selection_test_size"],
                        "path":get_output_model_selection_path(target, cell_line, task["name"], outer, inner)
                    } for inner in range(settings["selection_holdouts"])
                ]
            if len(jobs):
                with Pool(cpu_count()) as p:
                    list(tqdm(p.imap(kwarged_job, jobs), desc="Holdouts", leave=False, total=len(jobs)))