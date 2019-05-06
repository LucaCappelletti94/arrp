import glob
from typing import List


def get_cell_lines(target: str)->List[str]:
    return [
        e.split(".")[0].split("/")[-1].split("_")[0] for e in glob.glob(
            "{target}/regions/*.bed".format(target=target)
        )
    ]
