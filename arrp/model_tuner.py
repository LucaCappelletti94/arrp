import matplotlib
matplotlib.use('Agg')
import  matplotlib.pyplot as plt
from plot_keras_history import plot_history
from typing import Callable, Dict
from gaussian_process import GaussianProcess, Space
from plot_keras_history import plot_keras_history
from .model_fit import fit
from .model import model
import numpy as np
import pandas as pd

class ModelTuner:
    def __init__(self, structure:Callable, space:Space, holdouts:Callable, training:Dict, monitor:str):
        self._structure = structure
        self._space = space
        self._holdouts = holdouts
        self._training = training
        self._monitor = monitor
        self._iteration = 0

    def _score(self, **structure:Dict):
        """Return average score for given monitor key."""
        scores = []
        for i, ((training_set, testing_set), _) in enumerate(self._holdouts()):
            history = fit(training_set, testing_set, model(*self._structure(**structure)), self._training)
            path = "{cache}/{iteration}/{holdout}".format(
                cache=self._cache_dir,
                iteration=self._iteration,
                holdout=i
            )
            pd.DataFrame(history).to_csv("{path}/history.csv".format(path=path))
            plot_history(history)
            plt.savefig("{path}/history.png".format(path=path))
            scores.append(history[self._monitor][-1])
        self._iteration+=1
        return -np.exp(np.mean(scores))


    def tune(self, cache_dir:str, **kwargs)->Dict:
        self._cache_dir = cache_dir
        gp = GaussianProcess(self._score, self._space, cache_dir=cache_dir)
        gp.minimize(**kwargs)
        return gp.best_parameters