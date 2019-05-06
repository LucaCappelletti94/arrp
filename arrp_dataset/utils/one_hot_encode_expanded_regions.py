from tqdm import tqdm
from fasta_one_hot_encoder import FastaOneHotEncoder
from .get_cell_lines import get_cell_lines
import os


def one_hot_encode_expanded_regions(target: str, max_k: int=1):
    regions = get_cell_lines(target)
    print("One-hot encode nucleotides windows.")
    tqdm_bar = tqdm(total=max_k*len(regions), desc="One-hot encoding regions")
    os.makedirs(
        "{target}/one_hot_encoded_expanded_regions".format(target=target), exist_ok=True)
    for k in range(1, 1+max_k):
        encoder = FastaOneHotEncoder(
            nucleotides="acgtn",
            kmers_length=k,
            lower=True,
            sparse=False
        )
        for region in regions:
            path = "{target}/one_hot_encoded_expanded_regions/{region}_{k}_mers.csv".format(
                region=region,
                k=k,
                target=target
            )
            expand_region_path = "{target}/expanded_regions/{region}.fa".format(
                region=region,
                target=target
            )
            if not os.path.exists(path):
                encoder.transform_to_df(expand_region_path).to_csv(path)
            tqdm_bar.update()
    tqdm_bar.close()