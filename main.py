import numpy as np
from losses import mse

y_true = np.array([5, 10])
y_pred = np.array([3, 14])

print(mse(y_true, y_pred))
