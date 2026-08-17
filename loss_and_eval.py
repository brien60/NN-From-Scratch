import numpy as np
import math

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
    try: 
        return -math.log(a[np.argmax(y)][0])
    except Exception: 
        return -math.log(a[np.argmax(y)])

# L2 Regularization
def L2(model, optim):
    weight_decay = optim.weight_decay
    lr = optim.lr

    lambda_over_2n = weight_decay / (2 * lr) # (lambda*lr / n) * 1/(2*lr)

    weight_squared_sum = 0.0
    for layer in model.layers:
        for row in layer.W:
            for weight in row:
                weight_squared_sum += weight ** 2

    return lambda_over_2n * weight_squared_sum


def acc(y, a):
    return np.argmax(y) == np.argmax(a)