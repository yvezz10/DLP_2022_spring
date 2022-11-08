from torch import softmax
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, activate):
        super(EEGNet, self).__init__()

        activateFunction = {
            'ELU' : nn.ELU(alpha= 1.0),
            'ReLU' : nn.ReLU(),
            'LeakyReLU' : nn.LeakyReLU()
        }

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias= False),
            nn.BatchNorm2d(16, affine=True)
        )

        self.depthwiseCov = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activateFunction[activate],
            nn.AvgPool2d((1, 4), stride=(1,4), padding = 0),
            nn.Dropout(0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            activateFunction[activate],
            nn.AvgPool2d((1, 8), stride=(1, 8), padding = 0),
            nn.Dropout(0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features = 736, out_features = 2, bias=True),
        )

        
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseCov(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        output = self.classify(x)

        return output
         
class DeepConvNet(nn.Module):
    def __init__(self, activate):
        super(DeepConvNet, self).__init__()

        activateFunction = {
            'ELU' : nn.ELU(alpha= 1.0),
            'ReLU' : nn.ReLU(),
            'LeakyReLU' : nn.LeakyReLU()
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), stride=(1,2)),
            nn.Conv2d(25, 25, kernel_size=(2,1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            activateFunction[activate],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1,5), stride=(1,2)),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            activateFunction[activate],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1,5)),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            activateFunction[activate],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1,5)),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            activateFunction[activate],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.classify = nn.Sequential(
            nn.Linear(1600, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.classify(x)

        return output
