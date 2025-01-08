import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object. 
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()
        
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 定义第三个卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 定义全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 4 * 4)  # 展平操作
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
        # return torch.softmax(out)