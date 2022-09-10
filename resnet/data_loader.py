import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

data_path = "../train"
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
])
data = ImageFolder(root=data_path,transform=data_transforms)
train_size = int(len(data)*0.9)
test_size = len(data)-train_size
data_train,data_test = torch.utils.data.random_split(data,[train_size,test_size])
train_loader = torch.utils.data.DataLoader(data_train,batch_size=10,shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test,batch_size=10,shuffle=False)