from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 0.01
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 50
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = '../Part 1/data'

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
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def train():
    """
    Performs training and evaluation of CNN model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # 数据转换和加载
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=FLAGS.data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=FLAGS.data_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size,
                                             shuffle=False, num_workers=2)

    # 初始化模型、损失函数和优化器
    net = CNN(n_channels=3, n_classes=10)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=FLAGS.learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=FLAGS.learning_rate)

    # 记录每个batch的损失和准确率
    batch_losses = []
    batch_accuracies = []

    # 训练循环
    for epoch in range(FLAGS.max_steps):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())
            batch_accuracies.append(accuracy(outputs, labels))

            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # 每eval_freq次迭代进行一次评估
        if epoch % FLAGS.eval_freq == 0 or epoch == FLAGS.max_steps - 1:
            net.eval()
            test_accuracy = 0.0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = net(images)
                    test_accuracy += accuracy(outputs, labels)
            test_accuracy /= len(testloader)
            print(f'Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.4f}')
            net.train()

    print('Finished Training')

    # 绘制训练损失和准确率
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, label='Training Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss over Batches')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(batch_accuracies, label='Training Accuracy')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Batches')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()