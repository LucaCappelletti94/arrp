import subprocess
import os
from .get_cell_lines import get_cell_lines


def expand_regions(target: str, genome: str):
    """Expand the genomic regions using data withing given target and genome."""
    print("Expanding regions.")
    os.makedirs(
        "{target}/expanded_regions".format(target=target), exist_ok=True)
    for region in get_cell_lines(target):
        goal = "{target}/expanded_regions/{region}.fa".format(
            region=region,
            target=target
        )
        region_path = "{target}/regions/{region}.bed".format(
            region=region,
            target=target
        )
        if not os.path.exists(goal):
            subprocess.run(
                ["fastaFromBed", "-fi", "{target}/{genome}.fa".format(genome=genome, target=target), "-bed", region_path, "-fo", goal])
