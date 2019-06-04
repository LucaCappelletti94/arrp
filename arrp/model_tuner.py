import os
import pandas as pd
import numpy as np
from .model import model
from .model_fit import fit
from gaussian_process import GaussianProcess, Space
from typing import Callable, Dict


class ModelTuner:
    def __init__(self, structure: Callable, space: Space, holdouts: Callable, training: Dict):
        self._structure = structure
        self._space = space
        self._holdouts = holdouts
        self._training = training

    @classmethod
    def _calculate_score(cls, last_epoch: Dict) -> float:
        return last_epoch["val_auprc"] * (1 - last_epoch["val_loss"]) * last_epoch["val_acc"] * last_epoch["val_auroc"]

    def _score(self, **structure: Dict):
        """Return average model score."""
        return -np.mean([
            self._calculate_score({
                k: v[-1] for k, v in fit(training_set, testing_set, model(*self._structure(**structure)), self._training).items()
            }) for (training_set, testing_set), _ in self._holdouts()])

    def tune(self, cache_dir: str, **kwargs) -> Dict:
        self._cache_dir = cache_dir
        gp = GaussianProcess(self._score, self._space, cache_dir=cache_dir)
        gp.minimize(**kwargs)
        return gp.best_parameters
