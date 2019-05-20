from arrp import tasks_generator, validation_holdouts_generator, selection_holdouts_generator
from arrp.utils import tqdm, load_settings

def test_load():
    target = "test_dataset"
    settings = load_settings(target)
    for task in tqdm(list(tasks_generator(target)), desc="Jobs"):
        [
            _ for _ in tqdm(selection_holdouts_generator(*task), desc="Selection holdouts", total=settings["selection_holdouts"])
        ]
        [
            _ for _ in tqdm(validation_holdouts_generator(*task), desc="Validation holdouts", total=settings["validation_holdouts"])
        ]