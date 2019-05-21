from .get_cell_lines import get_cell_lines
import pandas as pd
import os
from .tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(classes, filename):
    encoder = OneHotEncoder(categories='auto', sparse=False)
    encoder.fit(classes.reshape(-1, 1))
    one_hot_encoded = encoder.transform(classes.reshape(-1, 1))
    return pd.DataFrame(
        one_hot_encoded, columns=encoder.categories_,
        dtype="int").to_csv(filename)


def one_hot_encode_classes(target: str):
    os.makedirs(
        "{target}/one_hot_encoded_classes".format(target=target), exist_ok=True)
    for region in tqdm(get_cell_lines(target), desc="One-hot encode classes"):
        region_classes = "{target}/classes/{region}.csv".format(
            region=region,
            target=target
        )
        path = "{target}/one_hot_encoded_classes/{region}.csv".format(
            region=region,
            target=target
        )
        if not os.path.exists(path):
            classes = pd.read_csv(region_classes, header=None)
            one_hot_encode(classes[0].ravel(), path)
