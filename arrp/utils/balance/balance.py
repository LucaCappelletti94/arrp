import numpy as np
from typing import Tuple, Dict


def balance_generic(array: np.ndarray, classes: np.ndarray, balancing_max: int, output: int)->Tuple:
    """Balance given arrays using given max and expected output class.
        arrays: np.ndarray, array to balance
        classes: np.ndarray, output classes
        balancing_max: int, maximum numbers per balancing maximum
        output: int, expected output class.
    """
    output_class_mask = classes == output
    retain_mask = classes != output
    n = np.sum(output_class_mask)
    if n > balancing_max:
        datapoints_to_remove = n - balancing_max
        mask = np.ones(shape=n)
        mask[:datapoints_to_remove] = 0
        np.random.shuffle(mask)
        output_class_mask[np.where(output_class_mask)] = mask
        array = array[np.logical_or(output_class_mask, retain_mask).reshape(-1)]
    return array

def umbalanced(*dataset_split:Tuple)->Tuple:
    """Leave data as they are."""
    return dataset_split


def balanced(*dataset_split:Tuple, balancing_max: int)->Tuple:
    """Balance training set using given balancing maximum.
        *dataset_split:Tuple, Tuple of arrays.
        balancing_max: int, balancing maximum.
    """
    assert isinstance(balancing_max, int)
    y_train = dataset_split[-2]

    balanced_dataset_split = []
    
    for i, array in enumerate(dataset_split):
        if i%2==0: # Balance only training data
            for output_class in [0, 1]:
                array = balance_generic(
                    array, y_train, balancing_max, output_class
                )
        balanced_dataset_split.append(array)
        
    return balanced_dataset_split


def full_balanced(*dataset_split:Tuple, balancing_max:int, rate: Tuple[int, int])->Tuple:
    """Balance training set using given balancing maximum.
        *dataset_split:Tuple, Tuple of arrays.
        balancing_max: int, balancing maximum.
        rate: Tuple[int, int], rates beetween the two classes.
    """
    assert isinstance(rate, tuple) and all(isinstance(v, int) for v in rate)
    dataset_split = balanced(*dataset_split, balancing_max=balancing_max)
    y_test = dataset_split[-1]
    balanced_dataset_split = []
    
    for i, array in enumerate(dataset_split):
        if i%2==1: # Balance only testing data
            for output_class in [0, 1]:
                opposite = 1 - output_class
                array = balance_generic(
                    array, y_test, int(np.sum(y_test == opposite)*rate[opposite]/rate[output_class]), output_class)
        balanced_dataset_split.append(array)
            
    return balanced_dataset_split

balancing_callbacks = {
    "umbalanced": umbalanced,
    "balanced": balanced,
    "full_balanced": full_balanced
}

def get_balancing_kwargs(balance_callback:str, positive_class:str, negative_class:str, settings:Dict):
    class_balancing = settings["class_balancing"]
    kwargs = {
        "umbalanced": {},
        "balanced": {
            "balancing_max": settings["max"]
        },
        "full_balanced": {
            "rate": (class_balancing.get(positive_class, 0), class_balancing.get(negative_class, 0)),
            "balancing_max": settings["max"]
        }
    }
    return kwargs[balance_callback]

def balance(dataset_split:Tuple, balance_callback:str, positive_class:str, negative_class:str, settings:Dict)->Tuple:
    global balancing_callbacks
    return balancing_callbacks[balance_callback](*dataset_split, **get_balancing_kwargs(balance_callback, positive_class, negative_class, settings))