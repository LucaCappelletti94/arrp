from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import glob
import shutil
import os
from .downloader import downloader


def merger(target: str, genome: str):
    with open('{target}/{genome}.fa'.format(target=target, genome=genome), 'wb') as wfd:
        for f in tqdm(glob.glob("{genome}/chromosomes/*.fa".format(genome=genome))):
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd, 1024*1024*10)


def genome_exists(target: str, genome: str)->bool:
    """Return a boolean representing if a file in expected genome position exists."""
    return os.path.exists("{target}/{genome}.fa".format(target=target, genome=genome)) or os.path.exists("{target}/expanded_regions".format(target=target))


def download_genome(target: str, genome: str):
    """Download given genome if it doesn't already exists in given target."""
    if genome_exists(target, genome):
        return
    print("Retrieving genome.")
    url = "http://hgdownload.cse.ucsc.edu/goldenPath/{genome}/chromosomes/chr{chr}.fa.gz"
    urls = [
        url.format(
            genome=genome,
            chr=chr
        ) for chr in list(range(1, 23)) + ["X", "Y"]
    ]

    print("Downloading genome {genome}.".format(genome=genome))
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(downloader, urls), total=len(urls)))

    print("Merging chromosomes.")
    merger(target, genome)

    print("Cleaning up.")
    shutil.rmtree(genome)
