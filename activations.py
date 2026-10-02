import numpy as np
x = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0]
])


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    x = np.array([-5, -2, 0, 2, 5])
    # An activation function doesn't really clean the dataset.
    return 1/(1 + np.exp(-x))
    # It transforms the numbers flowing through the neural network so
    # the network can learn complex patterns.


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


print(relu(x))
print(sigmoid(x))

print(softmax(x))
