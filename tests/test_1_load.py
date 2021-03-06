from arrp import tasks_generator,balanced_holdouts_generator
from auto_tqdm import tqdm

def test_load():
    target = "test_dataset"
    for task in tqdm(list(tasks_generator(target)), desc="Jobs"):
        for _, sub in balanced_holdouts_generator(*task)():
            for _, _ in sub():
                pass