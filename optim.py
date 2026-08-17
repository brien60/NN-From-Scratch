from helpers import *

class Optimizer:
    def __init__(self, lr, batch_size, weight_decay = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def __call__(self, layer): # update the parameters
        # print(f"{len(layer.W)}x{len(layer.W[0])}, {len(layer.W_grad)}x{len(layer.W_grad[0])}")
        if self.weight_decay is not None:
            layer.W = (1-self.weight_decay) * layer.W

        print(layer.W.shape, layer.b.shape)

        layer.W = np.subtract(layer.W, (self.lr/self.batch_size) * layer.dW)
        layer.b = np.subtract(layer.b, (self.lr/self.batch_size) * layer.dB)

        print(layer.W.shape, layer.b.shape)



