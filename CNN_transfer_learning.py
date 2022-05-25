import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pytorch_loader import ImagesLoader
import matplotlib.pyplot as plt


dataset = ImagesLoader()

train_split = 0.8
validation_split = 0.1
batch_size = 4

data_size = len(dataset)
print(f'dataset contains {data_size} Images')

train_size = int(train_split * data_size)
val_size = int(validation_split * data_size)
test_size = data_size - (val_size + train_size)
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_samples = DataLoader(val_data, batch_size=batch_size)
test_samples = DataLoader(test_data, batch_size=batch_size)

#%%
# print(dir(models.resnet50()))
class resnet50CNN(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.features = models.resnet50(pretrained=True).named_modules
        self.features.fc = torch.nn.Linear(2048, 13)
         # get the convolutional layers of resnet50. output size is 7x7x2048
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(7*7*2048, 4096), # first arg is the size of the flattened output from resnet50
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_size),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = F.relu(self.features(x))
        x = x.reshape(-1, 7*7*2048)
        x = self.regressor(x)
        return x

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad=False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad=True


cnn = resnet50CNN(13)
print(dir(cnn.features))

print(cnn.features)
#%%

def train_model(model, epochs):

    optimiser = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        for batch in train_samples:
            features, labels = batch
            predict = model(features)
            loss = F.cross_entropy(predict, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()


            if batch % 500 == 499:    # print every 50 mini-batches
               print(f'[{epoch + 1}, {batch + 1:5d}] loss: {loss}')
            

train_model(cnn, 5)
