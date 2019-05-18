from tqdm import tqdm
from fasta_one_hot_encoder import FastaOneHotEncoder
from .get_cell_lines import get_cell_lines
import os


def one_hot_encode_expanded_regions(target: str):
    regions = get_cell_lines(target)
    print("One-hot encode nucleotides windows.")
    os.makedirs(
        "{target}/one_hot_encoded_expanded_regions".format(target=target), exist_ok=True)
    encoder = FastaOneHotEncoder(
        nucleotides="acgtn",
        kmers_length=1,
        lower=True,
        sparse=False
    )
    for region in tqdm(regions, leave=False):
        path = "{target}/one_hot_encoded_expanded_regions/{region}.csv".format(
            region=region,
            target=target
        )
        if os.path.exists(path):
            continue
        expand_region_path = "{target}/expanded_regions/{region}.fa".format(
            region=region,
            target=target
        )
        if not os.path.exists(path):
            encoder.transform_to_df(expand_region_path).to_csv(path)