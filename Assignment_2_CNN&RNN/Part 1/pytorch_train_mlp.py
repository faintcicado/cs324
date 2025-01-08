from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    pred_labels = predictions.argmax(dim=1)
    true_labels = targets.argmax(dim=1)
    return (pred_labels == true_labels).float().mean().item()

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    batch_size = 64
    num_features = 10
    num_classes = 3
    x_data = torch.randn(batch_size, num_features)
    y_data = torch.zeros(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    y_data[range(batch_size), targets] = 1.0

    hidden_units = list(map(int, FLAGS.dnn_hidden_units.split(',')))
    model = MLP(n_inputs=num_features,
                n_hidden=hidden_units,
                n_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)

    for step in range(FLAGS.max_steps):
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = criterion(outputs, y_data.argmax(dim=1))
        loss.backward()
        optimizer.step()

        if step % FLAGS.eval_freq == 0:
            with torch.no_grad():
                acc = accuracy(outputs, y_data)
                print(f"Step: {step}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
    # YOUR TRAINING CODE GOES HERE


def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()