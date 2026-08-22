import math
import numpy as np

def matrix_mult(M, N):
    m, n, r = len(M), len(M[0]), len(N[0])
    assert n == len(N), f"{m}x{n} and {len(N)}x{r} matrices sizes incompatible"

    result = [[0 for _ in range(r)] for _ in range(m)]

    for i in range(m):
        for j in range(r):
            sum = 0
            for k in range(n):
                sum += M[i][k] * N[k][j]
            result[i][j] = sum
    return result

def matrix_scalar_mult(c, M):
    m, n = len(M), len(M[0])
    scaled = [[c * M[i][j] for j in range(n)] for i in range(m)]

    return scaled

def matrix_add(M, N):
    m, n = len(M), len(M[0])
    assert m == len(N) and n == len(N[0]), f"{m}x{n} and {len(N)}x{len(N[0])} matrices sizes incompatible"

    result = [[M[i][j] + N[i][j] for j in range(n)] for i in range(m)]
    return result

def matrix_sub(M, N):
    m, n = len(M), len(M[0])
    assert m == len(N) and n == len(N[0]), f"{m}x{n} and {len(N)}x{len(N[0])} matrices sizes incompatible"

    result = [[M[i][j] - N[i][j] for j in range(n)] for i in range(m)]
    return result

def hadamard(M, N):
    m, n = len(M), len(M[0])
    assert m == len(N) and n == len(N[0]), f"{m}x{n} and {len(N)}x{len(N[0])} matrices sizes incompatible"

    result = [[M[i][j] * N[i][j] for j in range(n)] for i in range(m)]
    return result

def vector_softmax(z):
    m, n = len(z), len(z[0])

    assert n == 1, f"This function only works on vectors, not a {m}x{n} matrix"
    result = [[0.0 for _ in range(n)] for _ in range(m)]

    sum = 0.0
    for i in range(m):
        for j in range(n):
            e = math.exp(z[i][j])
            result[i][j] = e
            sum += e

    return matrix_scalar_mult(1/sum, result)
        

def np_softmax(z):
    exp = np.exp(z)
    return exp / np.sum(exp, axis = 1, keepdims=True)



def transpose(M):
    m, n = len(M), len(M[0])
    # print(f"{m} x {n}")
    transposed = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(m):
        for j in range(n):
            transposed[j][i] = M[i][j]

    return transposed


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def vector_sigmoid(z):
    return [[sigmoid(z[i][0])] for i in range(len(z))]
    
def vector_dsigmoid(z):
    return [[dsigmoid(z[i][0])] for i in range(len(z))]

def np_sigmoid(z):
    result = np.zeros_like(z)
    for b in range(z.shape[0]):
        for i in range(z.shape[1]):
            result[b][i] = sigmoid(z[b][i])
    return result

def np_dsigmoid(z):
    result = np.zeros_like(z)
    for b in range(z.shape[0]):
        for i in range(z.shape[1]):
            result[b][i] = dsigmoid(z[b][i])
    return result

def list_dsigmoid(z):
    return np.array([dsigmoid(z[i]) for i in range(z.shape[0])])