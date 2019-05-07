import shutil
import os

def clear(target:str):
    [
        shutil.rmtree(
            "{target}/{directory}".format(target=target, directory=directory), ignore_errors=True
        ) for directory in [
            "expanded_regions",
            "one_hot_encoded_classes",
            "one_hot_encoded_expanded_regions"
        ]
    ]
    gzip_dir = "{target}/data".format(target=target)
    [
        os.remove(
            "{gzip_dir}/{document}".format(gzip_dir=gzip_dir, document=document)
        ) for document in os.listdir(gzip_dir) if document.endswith(".csv")
    ]