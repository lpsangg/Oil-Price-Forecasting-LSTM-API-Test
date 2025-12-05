import numpy as np
from model.dataloader import create_dataset

def test_create_dataset_shapes():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X, y = create_dataset(data, window=3)

    assert len(X) == len(y)
    assert X.shape[1] == 3  # window size
    assert X.shape[2] == 1  # feature dimension
    assert y.shape[1] == 1  # output dimension
