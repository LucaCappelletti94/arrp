from .utils import download_genome, expand_regions, one_hot_encode_regions, one_hot_encode_expanded_regions, ungzip_data

def build(target:str, genome:str="hg19"):
    download_genome(target, genome)
    ungzip_data(target)
    expand_regions(target, genome)
    one_hot_encode_regions(target)
    one_hot_encode_expanded_regions(target)