import numpy as np


y_true = np.array([5, 10])
y_pred = np.array([3, 14])


def mse(y_true, y_pred):
    pred_error = y_pred - y_true
    total = np.square(pred_error)
    return np.mean(total)
