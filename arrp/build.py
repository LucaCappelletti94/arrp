from .utils import download_genome, expand_regions, one_hot_encode_regions, one_hot_encode_expanded_regions, ungzip_data, load_settings
from .tasks import build_tasks

def build(target:str):
    settings = load_settings(target)
    genome = settings["genome"]
    download_genome(target, genome)
    ungzip_data(target)
    expand_regions(target, genome)
    one_hot_encode_regions(target)
    one_hot_encode_expanded_regions(target)
    build_tasks(
        target,
        settings["tasks"],
        settings["holdouts"],
        settings["validation_split"],
        settings["test_split"],
        settings["balance"]
    )