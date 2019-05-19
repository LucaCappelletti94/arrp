from .utils import mkdir


@mkdir
def get_cell_line_path(target:str, cell_line:str):
    return "{target}/cell_lines/{cell_line}".format(target=target, cell_line=cell_line)

@mkdir
def get_task_path(path:str, task:str):
    return "{path}/tasks/{task}".format(path=path, task=task.replace(" ", "_"))

@mkdir
def get_model_validation_path(path:str):
    return "{path}/model_validation".format(path=path)

@mkdir
def get_model_selection_path(path:str):
    return "{path}/model_selection".format(path=path)

@mkdir
def get_holdout_path(path:str, holdout:int):
    return "{path}/holdouts/{holdout}".format(path=path, holdout=holdout)

@mkdir
def get_input_path(path:str):
    return "{path}/input".format(path=path)

@mkdir
def get_output_path(path:str):
    return "{path}/output".format(path=path)

@mkdir
def get_test_path(path:str):
    return "{path}/test".format(path=path)

@mkdir
def get_train_path(path:str):
    return "{path}/train".format(path=path)

def get_epigenomic_data_path(path:str):
    return "{path}/epigenomic_data.csv".format(path=path)

def get_nucleotides_sequences_path(path:str):
    return "{path}/nucleotides_sequences.csv".format(path=path)

def get_classes_path(path:str):
    return "{path}/classes.csv".format(path=path)

def get_input_model_validation_path(target:str, cell_line:str):
    return get_model_validation_path(get_input_path(get_cell_line_path(target, cell_line)))

def get_input_model_selection_path(target:str, cell_line:str, holdout:int):
    return get_holdout_path(get_model_selection_path(get_input_path(get_cell_line_path(target, cell_line))), holdout)

def get_output_model_validation_path(target:str, cell_line:str, task:str):
    return get_model_validation_path(get_task_path(get_output_path(get_cell_line_path(target, cell_line)), task))

def get_output_model_selelection_path(target:str, cell_line:str, task:str, holdout:int):
    return get_holdout_path(get_model_selection_path(get_task_path(get_output_path(get_cell_line_path(target, cell_line)), task)), holdout)

def _build_csv_path(target:str, directory:str, cell_line:str):
    return "{target}/{directory}/{cell_line}.csv".format(
        target=target,
        directory=directory,
        cell_line=cell_line
    )

def get_raw_epigenomic_data_path(target:str, cell_line:str):
    return _build_csv_path(target, "epigenomic_data", cell_line)

def get_raw_nucleotides_sequences_path(target:str, cell_line:str):
    return _build_csv_path(target, "one_hot_encoded_expanded_regions", cell_line)

def get_raw_classes_path(target:str, cell_line:str):
    return _build_csv_path(target, "one_hot_encoded_classes", cell_line)