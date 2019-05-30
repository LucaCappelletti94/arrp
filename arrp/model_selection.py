from keras.models import Model
import os
import json
import pandas as pd
from skopt.callbacks import DeltaYStopper
from .utils import load_settings
from .load import balanced_holdouts_generator, tasks_generator
from .mlp import space as mlp_space
from .model import model
from .model_score import fit
from .model_tuner import ModelTuner
from gaussian_process import Space, TQDMGaussianProcess
from typing import Callable, Dict, Generator
from auto_tqdm import tqdm
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import matplotlib
from notipy_me import Notipy
matplotlib.use('Agg')

def dict_holdout_generator(generator)->Generator:
    if generator is None:
        return None

    def wrapper():
        for (training, testing), sub_generator in generator():
            yield (
                (({"mlp": training[0], "cnn": training[1]}), training[2]),
                (({"mlp": testing[0], "cnn": testing[1]}), testing[2])
            ), dict_holdout_generator(sub_generator)
    return wrapper


def get_path(name: str, target: str, cell_line: str, task: Dict, balance_mode: str):
    return "{target}/{name}/{cell_line}/{task}/{balance_mode}".format(
        target=target,
        name=name,
        cell_line=cell_line,
        task=task["name"].replace(" ", "_"),
        balance_mode=balance_mode
    )


def parameters_hash(best_parameters: str)->str:
    return hash(str(best_parameters))


def is_model_cached(path: str, best_parameters: Dict)->bool:
    hash_path = "{path}/hash.json".format(path=path)
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            return json.load(f)["hash"] == parameters_hash(best_parameters)
    return False


def save_results(model: Model, history: Dict, best_parameters: Dict, name:str, path: str, N:Notipy):
    os.makedirs(path, exist_ok=True)
    hash_path = "{path}/hash.json".format(path=path)
    with open(hash_path, "w") as f:
        json.dump({
            "hash": parameters_hash(best_parameters)
        }, f)
    model.save("{path}/model.h5".format(path=path))
    plot_history(history)
    plt.savefig("{path}/history.png".format(path=path))
    df = pd.DataFrame(history)
    df.to_csv("{path}/history.csv".format(path=path))
    with open("{path}/best_parameters.json".format(path=path), "w") as f:
        json.dump(best_parameters, f, indent=4)
    row = df[["auprc", "val_auprc"]].tail(1)
    row.index = [name]
    row.index.name = "Task name"
    N.add_report(row)


def collect_results(path: str, holdouts: int):
    pd.concat([
        pd.read_csv("{path}/{i}/history.csv".format(
            path=path,
            i=i
        ), index_col=0).tail(1) for i in range(holdouts)
    ]).mean().to_csv("{path}/average_results.csv".format(path=path))


def model_selection(target: str, name: str, structure: Callable, space: Space):
    with Notipy() as N:
        settings = load_settings(target)
        for task in tqdm(list(tasks_generator(target)), desc="Tasks"):
            generator = dict_holdout_generator(balanced_holdouts_generator(*task))
            path = get_path(name, *task)
            for i, ((training, testing), inner_holdouts) in enumerate(generator()):
                mlp_space["input_shape"] = (training[0]["mlp"].shape[1],)
                tuner = ModelTuner(structure, space, inner_holdouts,
                                settings["training"], settings["gaussian_process"]["monitor"])
                best_parameters = tuner.tune(
                    cache_dir="{path}/{i}/.gaussian_cache".format(path=path, i=i),
                    callback=[
                        TQDMGaussianProcess(
                            n_calls=settings["gaussian_process"]["tune"]["n_calls"]),
                        DeltaYStopper(
                            **settings["gaussian_process"]["early_stopping"])
                    ],
                    **settings["gaussian_process"]["tune"]
                )
                if not is_model_cached("{path}/{i}".format(path=path, i=i), best_parameters):
                    best_model = model(
                        *structure(**best_parameters),
                        metrics=["auprc", "auroc", "accuracy", "false_negatives", "false_positives",
                                "true_negatives", "true_positives", "precision", "recall"]
                    )
                    history = fit(training, testing, best_model, 
                                settings["training"])
                    save_results(best_model, history, best_parameters, task["name"],
                                "{path}/{i}".format(path=path, i=i), N)
            collect_results(path, settings["holdouts"]["quantities"][0])
