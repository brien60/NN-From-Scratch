from helpers import *

class Optimizer:
    def __init__(self, lr, batch_size, weight_decay = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def __call__(self, layer): # update the parameters
        # print(f"{len(layer.W)}x{len(layer.W[0])}, {len(layer.W_grad)}x{len(layer.W_grad[0])}")
        if self.weight_decay is not None:
            layer.W = matrix_scalar_mult(1-self.weight_decay, layer.W)

        layer.W = matrix_sub(layer.W, matrix_scalar_mult(self.lr/self.batch_size, layer.W_grad))
        layer.b = matrix_sub(layer.b, matrix_scalar_mult(self.lr/self.batch_size, layer.b_grad))



class OptimizerNP:
    def __init__(self, lr, batch_size, weight_decay = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def __call__(self, layer): # update the parameters
        # print(f"{len(layer.W)}x{len(layer.W[0])}, {len(layer.W_grad)}x{len(layer.W_grad[0])}")
        if self.weight_decay is not None:
            layer.W = (1-self.weight_decay) * layer.W

        layer.W = np.subtract(layer.W, (self.lr/self.batch_size) * layer.dW)
        layer.b = np.subtract(layer.b, (self.lr/self.batch_size) * layer.dB)



