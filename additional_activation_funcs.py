import numpy as np
def ReLU(x):
    return np.maximum(0,x)

def d_ReLU(x):
    return np.where(x <= 0, 0, 1)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def d_selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x > 0, 1, alpha * np.exp(x))