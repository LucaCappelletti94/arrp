import matplotlib
matplotlib.use('Agg')
import  matplotlib.pyplot as plt
from plot_keras_history import plot_history
from typing import Callable, Dict
from gaussian_process import GaussianProcess, Space
from .model_fit import fit
from .model import model
import numpy as np
import pandas as pd
import os

class ModelTuner:
    def __init__(self, structure:Callable, space:Space, holdouts:Callable, training:Dict):
        self._structure = structure
        self._space = space
        self._holdouts = holdouts
        self._training = training
        self._iteration = 0
        self._averages = None

    @classmethod
    def _calculate_score(cls, last_epoch:pd.DataFrame)->float:
        return last_epoch["val_auprc"] * (1 - last_epoch["val_loss"]) * last_epoch["val_acc"] * last_epoch["val_auroc"]

    def _score(self, **structure:Dict):
        """Return average model score."""
        scores = []
        averages = None
        for i, ((training_set, testing_set), _) in enumerate(self._holdouts()):
            history = fit(training_set, testing_set, model(*self._structure(**structure)), self._training)
            path = "{cache}/{iteration}/{holdout}".format(
                cache=self._cache_dir,
                iteration=self._iteration,
                holdout=i
            )
            os.makedirs(path, exist_ok=True)
            dfh = pd.DataFrame(history)
            dfh.to_csv("{path}/history.csv".format(path=path))
            plot_history(history)
            plt.savefig("{path}/history.png".format(path=path))
            plt.close()
            tail = dfh.tail(1)
            scores.append(self._calculate_score(tail))
            averages = tail if averages is None else pd.concat([
                tail, averages
            ])
        self._averages = averages.mean().to_frame().T if self._averages is None else pd.concat([
            self._averages, averages.mean().to_frame().T
        ])
        self._iteration+=1
        return -np.exp(np.mean(scores))


    def tune(self, cache_dir:str, **kwargs)->Dict:
        self._cache_dir = cache_dir
        gp = GaussianProcess(self._score, self._space, cache_dir=cache_dir)
        gp.minimize(**kwargs)
        if self._averages is not None:
            self._averages.to_csv("{path}/history.csv".format(path=self._cache_dir))
            plot_history({
                m:self._averages[m].values for m in self._averages.columns
            })
            plt.savefig("{path}/history.png".format(path=self._cache_dir))
            plt.close()
        return gp.best_parameters