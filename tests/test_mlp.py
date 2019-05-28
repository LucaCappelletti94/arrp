from arrp import build, mlp

def test_mlp():
    target = "test_dataset"
    build(target)
    mlp(target)