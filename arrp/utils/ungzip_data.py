from .ungzip import ungzip
from .get_cell_lines import get_cell_lines
import os

def ungzip_data(target: str):
    print("Expanding data.")
    for cell_line in get_cell_lines(target):
        path =  "{target}/epigenomic_data/{cell_line}.csv".format(
            target=target,
            cell_line=cell_line
        )
        if os.path.exists(path):
            continue
        ungzip("{path}.gz".format(path=path))