from torch.nn.modules.activation import Sigmoid

def sigmoid_smoothing(x):
    return 2 * Sigmoid(5 * x) - 1