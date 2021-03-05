import torch.nn as nn
import torch

class AuxiliaryClassifier(nn.Module):
    '''
    Auxiliary Classifier for a GoogLeNet-like CNN

    Attributes
    ----------
    in_planes : int
        Number of input feature maps
    num_classes: int
        Number of classes we need to classify
    '''
    
    def __init__(self, conv, num_classes=10):
        super(AuxiliaryClassifier, self).__init__()
        self.l1 = nn.AdaptiveAvgPool2d((1,1))
        self.l2 = nn.Conv2d(conv, 128, kernel_size=1)
        self.l3 = nn.Linear(128, 1024)
        self.l4 = nn.ReLU(True)
        self.l5 = nn.Linear(1024, num_classes)

    def forward(self, x):
        
        x = self.l1(x)
        x = self.l2(x)
        x = torch.flatten(x, 1)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        
        return x