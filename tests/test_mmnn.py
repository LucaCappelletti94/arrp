from arrp import build, mmnn

def test_mmnn():
    target = "test_dataset"
    build(target)
    mmnn(target)