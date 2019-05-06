from .utils import download_genome, expand_regions, one_hot_encode_regions, one_hot_encode_expanded_regions

def build(target:str, genome:str="hg19"):
    download_genome(target, genome)
    expand_regions(target, genome)
    one_hot_encode_regions(target)
    one_hot_encode_expanded_regions(target)