import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
from auto_tqdm import tqdm
from typing import Callable, Dict, Generator
from gaussian_process import Space, TQDMGaussianProcess
from .model_tuner import ModelTuner
from .model_score import fit
from .model import model
from .mlp import space as mlp_space
from .load import balanced_holdouts_generator, tasks_generator
from .utils import load_settings
from skopt.callbacks import DeltaYStopper
import pandas as pd
import json
import os
from keras.models import Model


def task_filter(target:str, cell_line:str, task:Dict, balance_mode:str)->bool:
    """Function to limit train to only some tasks of settings."""
    return cell_line == "GM12878" and task["name"] == "Active enhancers vs active promoters" and balance_mode == "umbalanced"

def filtered_task_generator(target:str):
    for task in tasks_generator(target):
        if task_filter(*task):
            yield task

def dict_holdout_generator(generator)->Generator:
    if generator is None:
        return None
    def wrapper():
        for (training, testing), sub_generator in generator():
            yield (
                (({"mlp":training[0], "cnn":training[1]}), training[2]),
                (({"mlp":testing[0], "cnn":testing[1]}), testing[2])
            ), dict_holdout_generator(sub_generator)
    return wrapper

def get_path(name:str, holdout:int, target:str, cell_line:str, task:Dict, balance_mode:str):
    return "{target}/{name}/{cell_line}/{task}/{balance_mode}/{holdout}".format(
        target=target,
        name=name,
        cell_line=cell_line,
        task=task["name"].replace(" ", "_"),
        balance_mode=balance_mode,
        holdout=holdout
    )

def save_results(model:Model, history:Dict, best_parameters:Dict, path:str):
    os.makedirs(path, exist_ok=True)
    model.save("{path}/model.h5".format(path=path))
    plot_history(history)
    plt.savefig("{path}/history.png".format(path=path))
    pd.DataFrame(history).to_csv("{path}/history.csv".format(path=path))
    with open("{path}/best_parameters.json".format(path=path), "w") as f:
        json.dump(best_parameters, f, indent=4)

def model_selection(target:str, name:str, structure:Callable, space:Space):
    settings = load_settings(target)
    for task in tqdm(list(filtered_task_generator(target)), desc="Tasks"):
        generator = dict_holdout_generator(balanced_holdouts_generator(*task))
        for i, ((training, testing), inner_holdouts) in enumerate(generator()):
            mlp_space["input_shape"] = (training[0]["mlp"].shape[1],)
            tuner = ModelTuner(structure, space, inner_holdouts, settings["training"], settings["gaussian_process"]["monitor"])
            best_parameters = tuner.tune(
                callback=[
                    TQDMGaussianProcess(n_calls=settings["gaussian_process"]["tune"]["n_calls"]),
                    DeltaYStopper(**settings["gaussian_process"]["early_stopping"])
                ],
                **settings["gaussian_process"]["tune"]
            )
            best_model = model(*structure(**best_parameters))
            history = fit(training, testing, best_model, settings["training"])
            save_results(best_model, history, best_parameters, get_path(name, i, *task))