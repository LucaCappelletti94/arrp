import pandas as pd
from typing import Dict
import numpy as np

def load_cellular_variables(target:str, cell_line:str):
    return pd.read_csv(
        "{target}/data/{cell_line}.csv".format(
            target=target,
            cell_line=cell_line
        ), 
        index_col=0
    )

def load_nucleotides_sequences(target:str, cell_line:str):
    return pd.read_csv(
        "{target}/one_hot_encoded_expanded_regions/{cell_line}.csv".format(
            target=target,
            cell_line=cell_line,
            k=1,
        ), 
        index_col=0
    ).values.reshape(-1, 200, 5)

def load_classes(target:str, cell_line:str):
    return pd.read_csv(
        "{target}/one_hot_encoded_classes/{cell_line}.csv".format(
            target=target,
            cell_line=cell_line,
        ), 
        index_col=0
    )

def get_active_regions(classes):
    return np.any([
        classes[column] for column in classes.columns if "A"==column.split("-")[0]
    ], axis=0)

def get_promoter_regions(classes):
    return np.any([
        classes[column] for column in classes.columns if "P"==column.split("-")[1]
    ], axis=0)

def load(target:str, cell_line:str, verbose:bool=True)->Dict:
    """Return a Dict containing four dataframes for the given cell line:
        1. Contains the cellular variables data
        2. Contains the nucleotides sequences.
        3. Contains the labels for active/inactive region.
        4. Contains the labels for promoter/enhancer region.
    """
    if verbose:
        print("Loading cellular variables.")
    cellular_variables = load_cellular_variables(target, cell_line)
    if verbose:
        print("Loading nucleotide sequences.")
    nucleotides_sequences = load_nucleotides_sequences(target, cell_line)
    if verbose:
        print("Loading classes.")
    classes = load_classes(target, cell_line)
    if verbose:
        print("Dropping unknown labels.")
    unknown = classes["UK"] == 1
    cellular_variables = cellular_variables.drop(index=cellular_variables.index[unknown])
    nucleotides_sequences = nucleotides_sequences[~unknown]
    classes = classes.drop(index=classes.index[unknown])
    classes = classes.drop(columns=["UK"])
    if verbose:
        print("Rendering labels.")
    active_regions = get_active_regions(classes)
    promoter_regions = get_promoter_regions(classes)

    return {
        "cellular_variables": cellular_variables,
        "nucleotides_sequences": nucleotides_sequences,
        "active_regions": active_regions,
        "promoter_regions": promoter_regions
    }