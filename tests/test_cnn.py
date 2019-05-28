from arrp import build, cnn

def test_cnn():
    target = "test_dataset"
    build(target)
    cnn(target)