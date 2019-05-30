import shutil
import os
from holdouts_generator import clear_holdouts_cache

def clear(target:str):
    [
        shutil.rmtree(
            "{target}/{directory}".format(target=target, directory=directory), ignore_errors=True
        ) for directory in [
            "one_hot_encoded_classes",
            "one_hot_encoded_expanded_regions",
            "cell_lines"
        ]
    ]
    gzip_dir = "{target}/epigenomic_data".format(target=target)
    [
        os.remove(
            "{gzip_dir}/{document}".format(gzip_dir=gzip_dir, document=document)
        ) for document in os.listdir(gzip_dir) if document.endswith(".csv")
    ]
    clear_holdouts_cache(".holdouts")