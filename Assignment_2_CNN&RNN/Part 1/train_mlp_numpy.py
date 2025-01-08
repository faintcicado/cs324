import argparse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from mlp_numpy import MLP  
from modules import CrossEntropy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500  # Adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 1  # Default is 1, which corresponds to SGD

test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
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


    X, y = make_moons(n_samples=1000,shuffle=True, noise=None, random_state=None)
    y_one_hot = np.eye(2)[y] 


    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    hidden_units = [int(x) for x in dnn_hidden_units.split(',')]
    n_inputs = X_train.shape[1]
    n_classes = y_train.shape[1]
    mlp = MLP(n_inputs, hidden_units, n_classes)
    loss_fn = CrossEntropy()

    n_samples = X_train.shape[0]

    
    for step in range(max_steps):

        # Mini-batch processing
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            predictions = mlp.forward(X_batch)

            loss = loss_fn.forward(predictions, y_batch)

            dout = loss_fn.backward(predictions, y_batch)
            mlp.backward(dout)

            # Update weights
            for layer in mlp.layers:
                if hasattr(layer, 'params'):
                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']

        # Evaluate on the test set at specified intervals
        if step % eval_freq == 0 or step == max_steps - 1:
            test_predictions = mlp.forward(X_test)
            test_loss = loss_fn.forward(test_predictions, y_test)
            test_accuracy = accuracy(test_predictions, y_test)
            # Train set evaluation
            train_predictions = mlp.forward(X_train)
            train_loss = loss_fn.forward(train_predictions, y_train)
            train_accuracy = accuracy(train_predictions, y_train)

            steps.append(step )
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # print(f"Step: {step}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Train Accuracy: {train_accuracy:.2f}%")
    
    print("Training complete!")
    draw_plot(batch_size)

  
def draw_plot(batch_size):

  # Plot the metrics
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(steps,test_losses, label='Test Loss')
    plt.plot(steps,train_losses, label='Train Loss')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(2, 1, 2)
    plt.plot(steps,test_accuracies, label='Test Accuracy')
    plt.plot(steps,train_accuracies, label='Train Accuracy')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    # plt.savefig('./figures/output_batchsize' + str(batch_size) + '.png') 
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
