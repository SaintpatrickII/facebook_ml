from re import M
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class resnet50CNN(torch.nn.Module):
#     def __init__(self, out_size):

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model = torchvision.models.resnet50(pretrained=True)
print(model)
# # model.fc = torch.nn.Sequential(nn.Linear(2048, 1024), # first arg is the size of the flattened output from resnet50
#             torch.nn.ReLU(),
#             torch.nn.Dropout(),
#             torch.nn.Linear(1024, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 13),
#             torch.nn.Softmax(dim=1)
#         )


# model.avgpool = Identity()

print(model)

#%%

def train_model(model, epochs):

    optimiser = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        for i, batch in enumerate(train_samples):
        # [train_samples, val_samples]:
        #     if batch == train_samples:
        #         model.train()
        #     else:
        #         model.eval()

            features, labels = batch
            predict = model(features)
            loss = F.cross_entropy(predict, labels)
            if epoch == train_samples:
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()


            if i % 500 == 499:    # print every 50 mini-batches
               print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
            

train_model(model, 5)
#%%
