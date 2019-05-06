from .ungzip import ungzip
from .get_cell_lines import get_cell_lines

def ungzip_data(target: str):
    print("Expanding data.")
    for cell_line in get_cell_lines(target):
        ungzip(
            "{target}/data/{cell_line}.csv.gz".format(
                target=target,
                cell_line=cell_line
            )
        )