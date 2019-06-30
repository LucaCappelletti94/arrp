import gc
from keras.backend import clear_session
from dict_hash import sha256
from notipy_me import Notipy
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
from auto_tqdm import tqdm
from typing import Callable, Dict, Generator
from gaussian_process import Space, TQDMGaussianProcess
from .model_tuner import ModelTuner
from .model_fit import fit
from .model import model
from .mlp import space as mlp_space
from holdouts_generator import clear_memory_cache
from .load import balanced_holdouts_generator, tasks_generator
from .utils import load_settings
from skopt.callbacks import DeltaYStopper
import pandas as pd
import json
import os
from keras.models import Model
import matplotlib
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


def parameters_hash(all_parameters: Dict)->str:
    return sha256(all_parameters)


def is_model_cached(path: str, all_parameters: Dict)->bool:
    hash_path = "{path}/hash.json".format(path=path)
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            return json.load(f)["hash"] == parameters_hash(all_parameters)
    return False


def save_results(model: Model, history: Dict, all_parameters:Dict, best_parameters: Dict, path: str, notifier: Notipy):
    os.makedirs(path, exist_ok=True)
    hash_path = "{path}/hash.json".format(path=path)
    with open(hash_path, "w") as f:
        json.dump({
            "hash": parameters_hash(all_parameters)
        }, f)
    model.save("{path}/model.h5".format(path=path))
    plot_history(history)
    plt.savefig("{path}/history.png".format(path=path))
    plt.close()
    df = pd.DataFrame(history)
    df.to_csv("{path}/history.csv".format(path=path))
    with open("{path}/best_parameters.json".format(path=path), "w") as f:
        json.dump(best_parameters, f, indent=4)
    notifier.add_report(df[["auprc", "val_auprc"]].tail(1))


def collect_results(path: str, holdouts: int):
    pd.concat([
        pd.read_csv("{path}/{i}/history.csv".format(
            path=path,
            i=i
        ), index_col=0).tail(1) for i in range(holdouts)
    ]).mean().to_csv("{path}/average_results.csv".format(path=path))


def model_selection(target: str, name: str, structure: Callable, space: Space):
    with Notipy() as notifier:
        settings = load_settings(target)
        for task in tqdm(list(tasks_generator(target)), desc="Tasks"):
            generator = dict_holdout_generator(
                balanced_holdouts_generator(*task))
            path = get_path(name, *task)
            for i, ((training, testing), inner_holdouts) in enumerate(generator()):
                mlp_space["input_shape"] = (training[0]["mlp"].shape[1],)
                model_space = Space({
                    "structure": space,
                    "training": settings["gaussian_process"]["training"],
                    "data_hash": {
                        "training": sha256(training),
                        "testing": sha256(testing)
                    }
                })
                all_parameters = {
                    "space":model_space,
                    "settings":settings
                }
                if not is_model_cached("{path}/{i}".format(path=path, i=i), all_parameters):
                    tuner = ModelTuner(structure, model_space, inner_holdouts)
                    best_parameters = tuner.tune(
                        cache_dir="{path}/{i}/.gaussian_cache".format(
                            path=path, i=i),
                        callback=[
                            TQDMGaussianProcess(
                                n_calls=settings["gaussian_process"]["tune"]["n_calls"]),
                            DeltaYStopper(
                                **settings["gaussian_process"]["early_stopping"])
                        ],
                        **settings["gaussian_process"]["tune"]
                    )
                    best_parameters["training"] = settings["training"]
                    best_model = model(
                        *structure(**best_parameters["structure"]),
                        metrics=["auprc", "auroc", "accuracy", "false_negatives", "false_positives",
                                 "true_negatives", "true_positives", "precision", "recall"]
                    )
                    history = fit(training, testing, best_model, settings["training"])
                    save_results(best_model, history, all_parameters, best_parameters,
                                 "{path}/{i}".format(path=path, i=i), notifier)
                    clear_session()
                del training
                del testing
                gc.collect()
            collect_results(path, settings["holdouts"]["quantities"][0])
            
