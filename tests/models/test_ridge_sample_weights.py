from datamodels import RidgeRegression
import numpy as np


def test_sample_weight():
    data = np.random.random((10000, 1, 4))
    x_train = data[:,:,:3]
    y_train = data[:,:,3:]

    ridge_full = RidgeRegression(sample_weight=np.ones(data.shape[0]))
    ridge_full.train(x_train, y_train)

    # Set second half of samples to zero
    data_size_half = int(data.shape[0] / 2)
    weight = np.concatenate((np.ones(data_size_half), np.zeros(data_size_half)))
    ridge_weighted = RidgeRegression(sample_weight=weight)
    ridge_weighted.train(x_train, y_train)
    # Check that with the weights the regression should provide different results
    assert(np.any(ridge_weighted.get_coef() != ridge_full.get_coef()))


if __name__ == "__main__":
    test_sample_weight()
