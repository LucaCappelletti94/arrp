import glob
from typing import List


def get_cell_lines(target: str)->List[str]:
    return [
        e.split(".bed")[0].split("/")[-1] for e in glob.glob(
            "{target}/regions/*.bed".format(target=target)
        )
    ]
