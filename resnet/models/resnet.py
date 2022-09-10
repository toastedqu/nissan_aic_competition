from torchvision import models
from torch import nn
 
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features,2)