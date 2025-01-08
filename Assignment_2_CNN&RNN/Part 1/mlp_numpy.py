from modules import * 

class MLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes the multi-layer perceptron object.
        
        This function should initialize the layers of the MLP including any linear layers and activation functions 
        you plan to use. You will need to create a list of linear layers based on n_inputs, n_hidden, and n_classes.
        Also, initialize ReLU activation layers for each hidden layer and a softmax layer for the output.
        
        Args:
            n_inputs (int): Number of inputs (i.e., dimension of an input vector).
            n_hidden (list of int): List of integers, where each integer is the number of units in each hidden layer.
            n_classes (int): Number of classes of the classification problem (i.e., output dimension of the network).
        """
        # Hint: You can use a loop to create the necessary number of layers and add them to a list.
        # Remember to initialize the weights and biases in each layer.
                # Store the layers
        self.layers = []
        
        # Input layer
        current_input = n_inputs
        
        # Hidden layers with ReLU activation
        for hidden_units in n_hidden:
            self.layers.append(Linear(current_input, hidden_units))
            self.layers.append(ReLU())
            current_input = hidden_units
        
        # Output layer followed by softmax
        self.layers.append(Linear(current_input, n_classes))
        self.layers.append(SoftMax())

    def forward(self, x):
        """
        Predicts the network output from the input by passing it through several layers.
        
        Here, you should implement the forward pass through all layers of the MLP. This involves
        iterating over your list of layers and passing the input through each one sequentially.
        Don't forget to apply the activation function after each linear layer except for the output layer.
        
        Args:
            x (numpy.ndarray): Input to the network.
            
        Returns:
            numpy.ndarray: Output of the network.
        """
        # Start with the input as the initial output
        out = x  # Start with the input as the initial output
        
        # Pass through each layer sequentially
        for layer in self.layers:
            out = layer.forward(out)
        
        return out

    def backward(self, dout):
        """
        Performs the backward propagation pass given the loss gradients.
        
        Here, you should implement the backward pass through all layers of the MLP. This involves
        iterating over your list of layers in reverse and passing the gradient through each one sequentially.
        You will update the gradients for each layer.
        
        Args:
            dout (numpy.ndarray): Gradients of the loss with respect to the output of the network.
        """
        # TODO: Implement the backward pass through each layer.
        # Hint: You will need to update 'dout' to be the gradient of the loss with respect to the input of each layer.
        
        # No need to return anything since the gradients are stored in the layers.
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
