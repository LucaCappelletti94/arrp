from arrp import tasks_generator,balanced_holdouts_generator
from arrp.utils import tqdm
from holdouts_generator import clear_holdouts_cache

def test_load():
    target = "test_dataset"
    for task in tqdm(list(tasks_generator(target)), desc="Jobs"):
        for _, sub in balanced_holdouts_generator(*task)():
            for _, _ in sub():
                pass
    clear_holdouts_cache(".holdouts")