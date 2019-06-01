from typing import Tuple, Dict
from keras.models import Model
from environments_utils import is_notebook
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
ktqdm = TQDMNotebookCallback if is_notebook() else TQDMCallback

def fit(training_set:Tuple, testing_set:Tuple, model:Model, training:Dict):
    """Return the model score for the given parameters.
        training_set:Tuple, datapoints to use for the training of the model.
        testing_set:Tuple, datapoints to use for the testing of the model.
        model:Model, the model to train.
        training:Dict, parameters for the training process.
    """
    return model.fit(
        *training_set,
        shuffle=True,
        verbose=0,
        validation_data=testing_set,
        callbacks=[
            ktqdm(),
            EarlyStopping(**training["early_stopping"]),
            ReduceLROnPlateau(**training["plateau"])
        ],
        **training["fit"]
    ).history