import argparse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from mlp_numpy import MLP as MLP_numpy
from modules import CrossEntropy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_mlp import MLP as MLP_pytorch

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500  # Adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 1  # Default is 1, which corresponds to SGD

test_losses_numpy = []
test_accuracies_numpy = []
train_losses_numpy = []
train_accuracies_numpy = []

test_losses_pytorch = []
test_accuracies_pytorch = []
train_losses_pytorch = []
train_accuracies_pytorch = []

steps = []

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    accuracy = np.mean(predicted_classes == true_classes) * 100
    return accuracy

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """

    X, y = make_moons(n_samples=1000, shuffle=True, noise=None, random_state=None)
    y_one_hot = np.eye(2)[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    hidden_units = [int(x) for x in dnn_hidden_units.split(',')]
    n_inputs = X_train.shape[1]
    n_classes = y_train.shape[1]

    # NumPy MLP
    mlp_numpy = MLP_numpy(n_inputs, hidden_units, n_classes)
    loss_fn_numpy = CrossEntropy()

    # PyTorch MLP
    mlp_pytorch = MLP_pytorch(n_inputs, hidden_units, n_classes)
    loss_fn_pytorch = nn.CrossEntropyLoss()
    optimizer_pytorch = optim.SGD(mlp_pytorch.parameters(), lr=learning_rate)

    n_samples = X_train.shape[0]

    for step in range(max_steps):

        # Mini-batch processing for NumPy MLP
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            predictions_numpy = mlp_numpy.forward(X_batch)
            loss_numpy = loss_fn_numpy.forward(predictions_numpy, y_batch)
            dout_numpy = loss_fn_numpy.backward(predictions_numpy, y_batch)
            mlp_numpy.backward(dout_numpy)

            # Update weights
            for layer in mlp_numpy.layers:
                if hasattr(layer, 'params'):
                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']

        # Mini-batch processing for PyTorch MLP
        for i in range(0, n_samples, batch_size):
            X_batch = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
            y_batch = torch.tensor(np.argmax(y_train[i:i + batch_size], axis=1), dtype=torch.long)

            optimizer_pytorch.zero_grad()
            predictions_pytorch = mlp_pytorch(X_batch)
            loss_pytorch = loss_fn_pytorch(predictions_pytorch, y_batch)
            loss_pytorch.backward()
            optimizer_pytorch.step()

        # Evaluate on the test set at specified intervals
        if step % eval_freq == 0 or step == max_steps - 1:
            # NumPy MLP evaluation
            test_predictions_numpy = mlp_numpy.forward(X_test)
            test_loss_numpy = loss_fn_numpy.forward(test_predictions_numpy, y_test)
            test_accuracy_numpy = accuracy(test_predictions_numpy, y_test)
            train_predictions_numpy = mlp_numpy.forward(X_train)
            train_loss_numpy = loss_fn_numpy.forward(train_predictions_numpy, y_train)
            train_accuracy_numpy = accuracy(train_predictions_numpy, y_train)

            test_losses_numpy.append(test_loss_numpy)
            test_accuracies_numpy.append(test_accuracy_numpy)
            train_losses_numpy.append(train_loss_numpy)
            train_accuracies_numpy.append(train_accuracy_numpy)

            # PyTorch MLP evaluation
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)
                test_predictions_pytorch = mlp_pytorch(X_test_tensor)
                test_loss_pytorch = loss_fn_pytorch(test_predictions_pytorch, y_test_tensor).item()
                test_accuracy_pytorch = accuracy(test_predictions_pytorch.numpy(), y_test)
                train_predictions_pytorch = mlp_pytorch(torch.tensor(X_train, dtype=torch.float32))
                train_loss_pytorch = loss_fn_pytorch(train_predictions_pytorch, torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)).item()
                train_accuracy_pytorch = accuracy(train_predictions_pytorch.numpy(), y_train)

            test_losses_pytorch.append(test_loss_pytorch)
            test_accuracies_pytorch.append(test_accuracy_pytorch)
            train_losses_pytorch.append(train_loss_pytorch)
            train_accuracies_pytorch.append(train_accuracy_pytorch)

            steps.append(step)

    print("Training complete!")
    draw_plot(batch_size)

def draw_plot(batch_size):
    # Plot the metrics
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(steps, test_losses_numpy, label='Test Loss (NumPy)')
    plt.plot(steps, train_losses_numpy, label='Train Loss (NumPy)')
    plt.plot(steps, test_losses_pytorch, label='Test Loss (PyTorch)')
    plt.plot(steps, train_losses_pytorch, label='Train Loss (PyTorch)')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(2, 1, 2)
    plt.plot(steps, test_accuracies_numpy, label='Test Accuracy (NumPy)')
    plt.plot(steps, train_accuracies_numpy, label='Train Accuracy (NumPy)')
    plt.plot(steps, test_accuracies_pytorch, label='Test Accuracy (PyTorch)')
    plt.plot(steps, train_accuracies_pytorch, label='Train Accuracy (PyTorch)')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size for training. If 1, training will be stochastic gradient descent.')
    FLAGS = parser.parse_args()
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.batch_size)

if __name__ == '__main__':

    start_time = time.time()

    main()

    end_time = time.time()
    running_time = end_time - start_time
    print("it took ", running_time, " seconds to run the code")