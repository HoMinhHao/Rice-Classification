import torch.nn as nn
from torch import flatten

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1=nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3)
        self.conv_layer2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer3=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2=nn.MaxPool2d(kernel_size=2,stride=1)
        # self.fc1=nn.Linear(1600,128)
        self.relu=nn.ReLU()
        # self.fc2=nn.Linear(128,num_classes)
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
    def forward(self,x):
        out=self.conv_layer1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv_layer2(out)
        out=self.bn1(out)
        out=self.max_pool1(out)
        out=self.conv_layer3(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv_layer4(out)
        out=self.bn2(out)
        out=self.max_pool2(out)
        
        # out=flatten(out,1)
        
        # out=self.fc1(out)
        # out=self.relu(out)
        # out=self.fc2(out)
        return out
        
        
