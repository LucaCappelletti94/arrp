from arrp import load_validation, holdouts_generator, tasks_generator
from arrp.utils import tqdm

def test_load():
    for task in tqdm(list(tasks_generator("test_dataset"))):
        load_validation(*task)
        [
            _ for _ in holdouts_generator(*task)
        ]