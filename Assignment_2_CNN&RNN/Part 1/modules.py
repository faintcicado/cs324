import numpy as np
class Linear:
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.params = {
            'weight': np.random.randn(in_features, out_features) * 0.01,
            'bias': np.zeros(out_features)
        }
        self.grads = {
            'weight': np.zeros_like(self.params['weight']),
            'bias': np.zeros_like(self.params['bias'])
        }
        self.input = None

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.input = x
        out = np.dot(x, self.params['weight']) + self.params['bias']
        return out

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        if self.input is None:
            raise ValueError("Input is None")
        
        # Compute gradients
        self.grads['weight'] = np.dot(self.input.T, dout)
        self.grads['bias'] = np.sum(dout, axis=0)
        
        # Compute gradient w.r.t. input for further backpropagation
        dx = np.dot(dout, self.params['weight'].T)
        return dx


class ReLU(object):
    def __init__(self):
        self.input = None

    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        """
        self.input = x  # Store input for backward pass
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        Gradient is 1 for x > 0, otherwise 0.
        """
        dx = dout * (self.input > 0)
        return dx


class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j.
        Uses the Max Trick for numerical stability.
        """
        # Max trick for stability: subtract max for each sample
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        """
        # Placeholder for combined backward calculation with CrossEntropy
        return dout


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        """
        # Clip values to prevent log(0)
        self.input = x
        self.y = y
        log_x = np.log(np.clip(x, 1e-10, 1.0))
        return -np.sum(y * log_x) / x.shape[0]

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        return (x - y) / x.shape[0]

