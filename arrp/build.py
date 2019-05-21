from .utils import download_genome, expand_regions, one_hot_encode_classes, one_hot_encode_expanded_regions, ungzip_data, load_settings
from .input import build_input
from .output import build_output
from .sanitize import sanitize

def build(target:str):
    settings = load_settings(target)
    genome = settings["genome"]
    download_genome(target, genome)
    ungzip_data(target)
    expand_regions(target, genome)
    one_hot_encode_classes(target)
    one_hot_encode_expanded_regions(target)
    sanitize(target)
    build_input(target, settings)
    build_output(target, settings)