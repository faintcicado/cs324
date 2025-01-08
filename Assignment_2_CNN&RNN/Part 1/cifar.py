import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pytorch_mlp import MLP as MLP_pytorch

# 定义数据转换
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        super(MLP, self).__init__()
        layers = []
        in_features = n_inputs
        for hidden_size in n_hidden:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, n_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        return self.layers(x)

# 定义损失函数和优化器
def train_model(net, trainloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

def test_model(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

def main():
    hidden_units = [512, 256, 128]
    n_inputs = 3 * 32 * 32
    n_classes = 10
    net = MLP(n_inputs, hidden_units, n_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_model(net, trainloader, criterion, optimizer, epochs=10)
    test_model(net, testloader)

if __name__ == '__main__':
    main()