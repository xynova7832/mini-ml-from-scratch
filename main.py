import numpy as np
from losses import mse, cce

y_true = np.array([5, 10])
y_pred = np.array([3, 14])

print(mse(y_true, y_pred))

correct_prob = np.array([0.8, 0.6, 0.9])

print(cce(correct_prob))


print(cce(np.array([0.8, 0.6, 0.9])))
print(cce(np.array([0.99, 0.99, 0.99])))
print(cce(np.array([0.0, 0.5, 1.0])))
