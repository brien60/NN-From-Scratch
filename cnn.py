import numpy as np
import math

from helpers import *
from loss_and_eval import *
from optim import *



class Layer:
    def __init__(self, input_neurons, output_neurons, softmax = False, output_layer = False):
        self.j = output_neurons
        self.k = input_neurons

        self.W = np.random.normal(scale = 1/math.sqrt(self.j), size = (self.j, self.k))
        self.b = np.random.normal(size = self.j)

        self.activations = None
        self.weighted_inputs = None

        self.softmax = softmax
        self.output_layer = output_layer

        self.clear_grad()

    def __call__(self, prev_activations):
        return self.forward(prev_activations)


    def forward(self, prev_activations):
        z = np.add(self.W @ prev_activations.reshape(self.k), self.b) # (j, ) 

        if self.softmax: a = np_softmax(z)
        else: a = np_sigmoid(z)

        self.activations = a
        self.weighted_inputs = z

        return a # (j, )
        


    def backward(self, prev_activations, y = None, next_W_T = None, next_dZ = None):
        if self.output_layer:
            dZ = np.subtract(self.activations, y) # BP1, BCE, NLL
            # print(f"output layer: {dZ.shape}")

        else:
            dZ = (next_W_T @ next_dZ) * np_dsigmoid(self.weighted_inputs) # BP2
            
        # BP3
        dW = dZ.reshape(-1, 1) @ prev_activations.reshape(self.k, 1).T


        # BP4
        self.dW = np.add(self.dW, dW)
        self.dB = np.add(self.dB, dZ)

        return dZ


    def clear_grad(self):
        self.dW = np.zeros(shape = (self.j, self.k))
        self.dB = np.zeros(shape = self.j)



class Convolution:
    def __init__(self, filters, channels, kernel_size, input_size, stride = 1):
        self.filters = filters
        self.channels = channels
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.stride = stride

        self.output_size = (self.input_size - self.kernel_size) // self.stride + 1

 
        self.W = np.random.normal(
            size = (self.filters, self.channels, self.kernel_size, self.kernel_size)
        )
        self.b = np.random.normal(
            size = self.filters
        )

        self.weighted_inputs = None
        self.clear_grad()


    def __call__(self, x):
            return self.forward(x)

    
    def clear_grad(self):
        self.dW = np.zeros(shape = (self.filters, self.channels, self.kernel_size, self.kernel_size))
        self.dB = np.zeros(shape = self.filters)

    

    # def forward(self, x):
    #     # x: (channels, length, width)
        
    #     f_maps = np.zeros(shape=(self.filters, self.output_size, self.output_size))
    #     weighted_inputs = np.zeros_like(f_maps)

    #     # hard-coded
    #     for f in range(self.filters):
    #         f_map = np.zeros(shape=(self.output_size, self.output_size))
    #         weighted_input = np.zeros_like(f_map)

    #         for i in range(self.output_size):
    #             for j in range(self.output_size):
    #                 z = self.b[f]

    #                 for u in range(self.kernel_size):
    #                     for v in range(self.kernel_size):
    #                         for c in range(self.channels):
    #                             input_channel = x[c]
    #                             W = self.W[f, c, :, :] # (kernel_size, kernel_size)

    #                             row = self.stride * i + u
    #                             col = self.stride * j + v
    #                             z += input_channel[row][col] * W[u][v]


    #                 f_map[i][j] = sigmoid(z)
    #                 weighted_input[i][j] = z

    #         f_maps[f] = f_map
    #         weighted_inputs[f] = weighted_input

    #     self.weighted_inputs = weighted_inputs

    #     return f_maps


    # matrix implementation
    def forward(self, x):
        W_matrix =  self.W.reshape(self.filters, -1)
        patches_matrix = np.zeros((self.channels * self.kernel_size ** 2, self.output_size ** 2))

        p = 0
        for i in range(self.output_size):
            for j in range(self.output_size):
                start_i, start_j = self.stride*i, self.stride*j
                offset = self.kernel_size
                patches_matrix[:, p] = x[:, start_i : start_i + offset, start_j : start_j + offset].reshape(-1)

                p+=1

        self.weighted_inputs = (W_matrix @ patches_matrix).reshape(self.filters, self.output_size, self.output_size)
        f_maps = np_sigmoid(self.weighted_inputs.reshape(-1)).reshape(self.filters, self.output_size, self.output_size)

        return f_maps


            


    def backward(self, prev_activations, dA):

        dZ = np.zeros(shape = (self.filters, self.output_size, self.output_size))
        prev_dA = np.zeros(shape = (self.channels, self.input_size, self.input_size))


        for f in range(self.filters):
            for i in range(self.output_size):
                for j in range(self.output_size):
                    dZ[f][i][j] = dA[f][i][j] * dsigmoid(self.weighted_inputs[f][i][j]) # calculcate dZ
                    self.dB[f] += dZ[f][i][j] # update bias gradient

                    for u in range(self.kernel_size):
                        for v in range(self.kernel_size):
                            for c in range(self.channels): 
                                W = self.W[f, c, :, :]
                                row = self.stride * i + u
                                col = self.stride * j + v

                                self.dW[f][c][u][v] += dZ[f][i][j] * prev_activations[c][row][col] # update weight gradient
                                prev_dA[c][row][col] += dZ[f][i][j] * W[u][v] # calculate dA for the previous layer
 

        return prev_dA

    

         


class MaxPool:
    def __init__(self, f_maps, input_size, pool_size):
        self.f_maps = f_maps
        self.input_size = input_size
        self.pool_size = pool_size

        assert self.input_size % self.pool_size == 0, f"Pool size {self.pool_size} is not valid for {self.input_size}x{self.input_size} feature maps"
        self.output_size = self.input_size // self.pool_size

        self.dZ_prev_dA = np.zeros(shape = (self.f_maps, self.input_size, self.input_size))


    def __call__(self, f_maps):
        return self.forward(f_maps)

    def forward(self, f_maps):
        pooled_f_maps = np.zeros(shape = (self.f_maps, self.output_size, self.output_size))

        for f in range(self.f_maps):
            f_map = f_maps[f]
            pooled_f_map = np.zeros(shape = (self.output_size, self.output_size))

            for i in range(self.output_size):
                for j in range(self.output_size):

                    a = float("-inf")
                    max_i, max_j = 0, 0

                    for u in range(self.pool_size):
                        for v in range(self.pool_size):
                            row = i * self.pool_size + u
                            col = j * self.pool_size + v

                            if f_map[row][col] > a:
                                a = f_map[row][col]
                                max_i, max_j = row, col

                    pooled_f_map[i][j] = a
                    self.dZ_prev_dA[f][max_i][max_j] = 1.0
                    


            pooled_f_maps[f] = pooled_f_map
        return pooled_f_maps


    

    def backward(self, next_W_T, next_dZ):
        dZ = np.reshape(next_W_T @ next_dZ, (self.f_maps, self.output_size, self.output_size))
        prev_dA = np.zeros(shape = (self.f_maps, self.input_size, self.input_size)) 

        for f in range(self.f_maps):
            for i in range(self.output_size):
                for j in range(self.output_size):
                    for u in range(self.pool_size):
                        for v in range(self.pool_size):
                            row = i * self.pool_size + u
                            col = j * self.pool_size + v

                            prev_dA[f][row][col] = dZ[f][i][j] * self.dZ_prev_dA[f][row][col] # calc dA of previous layer

        return prev_dA
        
                    



class CNN:
    def __init__(self, filters_list):

        self.conv1 = Convolution(filters=filters_list[0], channels=1, kernel_size=5, input_size=28)
        self.pool1 = MaxPool(f_maps=filters_list[0], input_size=self.conv1.output_size, pool_size=2)
        
        self.layer1 = Layer(
            input_neurons=self.pool1.f_maps * self.pool1.output_size ** 2,
            output_neurons=100,
        )
        self.classifier = Layer(
            input_neurons=100, output_neurons=10,
            output_layer=True, softmax=True
        )

        self.layers = [self.conv1, self.pool1, self.layer1, self.classifier]
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.activations_list = []
        self.x = x
        
        output = x
        for layer in self.layers:
            output = layer(output)
            self.activations_list.append(output)

        return output

    def backward(self, y):
        L = len(self.layers)
        for l in range(L-1, -1, -1): # go from L-1 to 0
            layer = self.layers[l]

            if (l-1 < 0): prev_activations = self.x
            else: prev_activations = self.activations_list[l-1]

            if type(layer) == Layer: 
                if layer.output_layer:
                    next_dZ = layer.backward(prev_activations, y=y)
                else:
                    next_dZ = layer.backward(prev_activations, next_W_T=self.layers[l+1].W.T, next_dZ=next_dZ)

            if type(layer) == MaxPool:
                prev_dA = layer.backward(self.layers[l+1].W.T, next_dZ)

            if type(layer) == Convolution:
                prev_dA = layer.backward(prev_activations, prev_dA)

        
    def step(self, optim):
        for layer in self.layers:
            if type(layer) != MaxPool: optim(layer)
    
    def clear_grad(self):
        for layer in self.layers:
            if type(layer) != MaxPool: layer.clear_grad()
        




