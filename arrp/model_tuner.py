import os
import pandas as pd
import numpy as np
from .model import model
from .model_fit import fit
from gaussian_process import GaussianProcess, Space
from typing import Callable, Dict
from keras.backend import clear_session
from dict_hash import sha256
import gc

class ModelTuner:
    def __init__(self, structure: Callable, space: Space, holdouts: Callable):
        self._structure = structure
        self._space = space
        self._holdouts = holdouts

    @classmethod
    def _calculate_score(cls, last_epoch: Dict) -> float:
        return last_epoch["val_auprc"] * last_epoch["val_auroc"] / last_epoch["val_loss"]

    def _score(self, structure: Dict, training:Dict, data_hash:Dict)->float:
        """Return average model score."""
        compiled_model = model(*self._structure(**structure))
        weights = compiled_model.get_weights()
        scores = []
        parameters_hash = sha256({
            "structure":structure,
            "training":training,
            "data_hash":data_hash
        })
        for (training_set, testing_set), _ in self._holdouts():
            path = "{cache_dir}/{parameters_hash}/{holdout_hash}".format(
                cache_dir=self._cache_dir,
                parameters_hash=parameters_hash,
                holdout_hash=sha256({
                    "training_set":training_set,
                    "testing_set":testing_set
                })
            )
            history_path = "{path}/history.csv".format(path=path)
            if not os.path.exists(history_path):
                clear_session()
                os.makedirs(path, exist_ok=True)
                compiled_model.set_weights(weights)
                history = pd.DataFrame(fit(training_set, testing_set, compiled_model, training))
                history.index.name = "Epochs"
                history.to_csv(history_path)
                clear_session()
            history = pd.read_csv(history_path, index_col="Epochs")
            scores.append(self._calculate_score(history.iloc[-1].to_dict()))
            del training_set
            del testing_set
            gc.collect()
        return -np.mean(scores)

    def tune(self, cache_dir: str, **kwargs) -> Dict:
        self._cache_dir = cache_dir
        gp = GaussianProcess(self._score, self._space, cache=False)
        gp.minimize(**kwargs)
        return gp.best_parameters
