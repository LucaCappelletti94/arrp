import os
import pandas as pd
import numpy as np
from .model import model
from .model_fit import fit
from gaussian_process import GaussianProcess, Space
from typing import Callable, Dict
from keras.backend import clear_session

class ModelTuner:
    def __init__(self, structure: Callable, space: Space, holdouts: Callable, training: Dict):
        self._structure = structure
        self._space = space
        self._holdouts = holdouts
        self._training = training

    @classmethod
    def _calculate_score(cls, last_epoch: Dict) -> float:
        return last_epoch["val_auprc"] * (1 - last_epoch["val_loss"]) * last_epoch["val_acc"] * last_epoch["val_auroc"]

    def _score(self, **structure: Dict)->float:
        """Return average model score."""
        compiled_model = model(*self._structure(**structure))
        weights = compiled_model.get_weights()
        scores = []
        for (training_set, testing_set), _ in self._holdouts():
            compiled_model.set_weights(weights)
            scores.append(self._calculate_score({
                k: v[-1] for k, v in fit(training_set, testing_set, compiled_model, self._training).items()
            }))
        clear_session()
        return -np.mean(scores)

    def tune(self, cache_dir: str, **kwargs) -> Dict:
        self._cache_dir = cache_dir
        gp = GaussianProcess(self._score, self._space, cache_dir=cache_dir)
        gp.minimize(**kwargs)
        return gp.best_parameters
