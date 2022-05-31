#%%
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

train_split = 0.2
validation_split = 0.1
batch_size = 12

data_size = len(dataset)
print(f'dataset contains {data_size} Images')

train_size = int(train_split * data_size)
val_size = int(validation_split * data_size)
test_size = data_size - (val_size + train_size)
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_samples = DataLoader(val_data, batch_size=batch_size)
test_samples = DataLoader(test_data, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        for i, param in enumerate(self.features.parameters()):
            if i < 47:
                param.requires_grad=False
            else:
                param.requires_grad=True
        self.features.fc = nn.Sequential(
            nn.Linear(2048, 1024), # first arg is the size of the flattened output from resnet50
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 13)
            )


    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        # x = torch.nn.Linear(256, 13),
        # x = torch.nn.Softmax(dim=1)
        # predict = self.fc(x)
        # fully_connected = x
        # print(x)
        return x

model = CNN()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#%%

def train_model(model, epochs):
    writer = SummaryWriter()
    model.train()
    optimiser = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_samples):
        # [train_samples, val_samples]:
        #     if batch == train_samples:
        #         model.train()
        #     else:
        #         model.eval()

            predict = model(features)
            loss = F.cross_entropy(predict, labels)
            # if epoch == train_samples:
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            writer.add_scalar('Loss', loss, i)
            writer.flush()
            if i % 100 == 99:   
                # print(batch) # print every 50 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
            

train_model(model, 5)



def check_accuracy(loader, model):
    if loader == train_samples:
        model.train()
        print('Checking accuracy on training set')
    else:
        print('Checking accuracy on test set')
        model.val()
    num_correct = 0
    num_samples = 0
    #   tells model not to compute gradients
    with torch.no_grad():
        for feature, label in loader:
            feature = feature.to(device)  # move to device
            label = label.to(device)
            scores = model(feature)
            # print(scores.max(1))
            _, preds = scores.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc}')



check_accuracy(train_samples, model)
check_accuracy(test_samples, model)




#%%