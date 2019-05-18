import shutil
import gzip
import requests
import os


def ungunzip(path):
    goal = path.split(".gz")[0]
    with gzip.open(path, 'rb') as f_in, open(goal, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def downloader(url: str):
    path = "/".join(url.split("/")[-3:])
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(requests.get(url).content)
    ungunzip(path)
    os.remove(path)
