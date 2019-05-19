from arrp import load_validation, holdouts_generator, tasks_generator
from arrp.utils import tqdm, load_settings

def test_load():
    target = "test_dataset"
    holdouts = load_settings(target)["holdouts"]
    for task in tqdm(list(tasks_generator(target)), desc="Jobs"):
        load_validation(*task)
        [
            _ for _ in tqdm(holdouts_generator(*task), desc="Holdouts", total=holdouts)
        ]