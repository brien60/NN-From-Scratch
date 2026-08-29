import numpy as np
import math
from cnn import MaxPool

# quadratic loss
def quadratic_loss(y, a):
    loss = 0.0
    for i in range(len(y)):
        loss += (y[i][0] - a[i][0]) ** 2
    loss /= 2
    return loss

# binary cross entropy loss
def BCE(y, a):
    loss = 0.0
    for i in range(len(y)):
        loss -= y[i][0]*math.log(a[i][0]) + (1 - y[i][0])*math.log((1 - a[i][0]))
    return loss 

# negative log-likelihood loss
def NLL(y, a):
    return -math.log(a[np.argmax(y)][0])

def batch_NLL(y, a):
    b = y.shape[0]
    # probs = a[np.arange(b), np.argmax(y, axis=1)]
    # return np.sum(-np.log(np.maximum(probs, 1e-15)))

    return np.sum(-np.log(a[np.arange(b), np.argmax(y, axis=1)]))

# L2 Regularization
def L2(model, optim):
    weight_decay = optim.weight_decay
    lr = optim.lr

    lambda_over_2n = weight_decay / (2 * lr) # (lambda*lr / n) * 1/(2*lr)

    weight_squared_sum = 0.0
    for layer in model.layers:
        if type(layer) != MaxPool:
            weight_squared_sum += np.sum(np.array(layer.W) ** 2)

    return lambda_over_2n * weight_squared_sum


def batch_acc(y, a):
    return np.sum(np.argmax(y, axis=1) == np.argmax(a, axis=1))


def acc(y, a):
    return np.argmax(y) == np.argmax(a)