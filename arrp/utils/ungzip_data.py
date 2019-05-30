from .ungzip import ungzip
import os
from typing import Dict

def ungzip_data(target: str, settings:Dict):
    print("Expanding data.")
    for cell_line in settings["cell_lines"]:
        path =  "{target}/epigenomic_data/{cell_line}.csv".format(
            target=target,
            cell_line=cell_line
        )
        if os.path.exists(path):
            continue
        ungzip("{path}.gz".format(path=path))