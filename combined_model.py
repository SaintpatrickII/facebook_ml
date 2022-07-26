# %%
from tqdm import tqdm
from combined_dataloader import ImageTextDataloader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from pytorch_loader import ImagesLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms


products_df = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/combined_final_dataset.csv'
image_folder = '/Users/paddy/Desktop/AiCore/facebook_ml/images_for_combined/'


# validation_split = 0.15
# batch_size = 32
# shuffle_dataset = True
# random_seed = 42

# dataset = ImageTextDataloader(Image_dir=image_folder, csv_file=products_df, transform=None)
# dataset_size = len(dataset)
# print(dataset[4000])
# # print(dataset_size)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

# train_samples = torch.utils.data.DataLoader(ImageTextDataloader, batch_size=batch_size, 
#                                            sampler=train_sampler)
# val_samples = torch.utils.data.DataLoader(ImageTextDataloader, batch_size=batch_size,
#                                                 sampler=valid_sampler)




# %%

def get_default_device():
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()




class TextClassifier(torch.nn.Module):
    def __init__(self, pretrained_weights=None):
        super().__init__()
        no_words = 26888
        embedding_size = 100
        self.embedding = torch.nn.Embedding(no_words, embedding_size)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(embedding_size, 32, 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 2),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(6272, 128)
        )


    '''
    TextClassifier Initiliser:

    We have an embedding layer, which is a matrix of size 26888x100, which is the size of our
    vocabulary. from here a TextClassifier is built using dropout & ReLU to avoid overfitting
    '''

    def forward(self, X):
        
        return self.layers(self.embedding(X))


    """
    forward: 

    The function takes in a batch of sentences, passes them through the embedding layer, and then
    passes them through the layers of the model
    :param X: the input data
    :return: The output of the last layer of the network.
    """


class ImageTextClassifier(nn.Module):
    def __init__(self):
        super(ImageTextClassifier, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        self.text_model = TextClassifier()
        self.main = nn.Sequential(nn.Linear(256, 13))
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
            torch.nn.Linear(512, 128)
            # torch.nn.ReLU(),
            # torch.nn.Linear((128), 13)
            )


    def forward(self, image_features, text_features):
        image_features = self.features(image_features)
        image_features = image_features.reshape(image_features.shape[0], -1)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)



        # x = torch.nn.Softmax(dim=1)(x)
        return combined_features

 
model = ImageTextClassifier()
model.to(device)


dataset = ImageTextDataloader()
dataloader = dataloader = torch.utils.data.DataLoader(dataset, batch_size=32 ,shuffle=True, num_workers=1)
# print(dataset[5000])



def train_model(model, epochs, optimiser):
# scheduler
    writer = SummaryWriter()
    print('training model')
    # dataset = ImageTextDataloader(Image_dir=image_folder, csv_file=products_df, transform=None)
    dataset_ite = tqdm(enumerate(dataloader))
    # optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        model.train()
        for phase in dataset_ite:
            model.train()
            #   if phase == train_samples:
            #     model.train()
            #     print('training')
            #   else:
            #     model.eval()
            #     print('val')
            #     print(phase)

        for i, (image_features, text_features, labels) in enumerate(phase):
        # with torch.set_grad_enabled(phase == train_samples):
        #   optimiser.zero_grad()
            num_correct = 0
            num_samples = 0
            features, labels = (image_features, text_features), labels
            features = features.to(device)  # move to device
            labels = labels.to(device)
            predict = model(features)
            labels = labels
            loss = F.cross_entropy(predict, labels)
            
            
            _, preds = predict.max(1)
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            loss.backward()
            optimiser.step()
            
            # writer.add_scalar('Loss', loss, epoch)
            # writer.add_scalar('Accuracy', acc, epoch)
            
            if i % 10 == 9:
                writer.add_scalar('Training Loss', loss, epoch)
                writer.add_scalar(' Training Accuracy', acc, epoch)
                print('training_loss')
            # else:
            #     writer.add_scalar('Validation Loss', loss, epoch)
            #     writer.add_scalar('Validation Accuracy', acc, epoch)
            #     print('val_loss') 
            # print(batch) # print every 50 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.5f}')
            print(f'Got {num_correct} / {num_samples} with accuracy: {(acc * 100):.2f}%')
            writer.flush()
        
              
              
              
            #   if scheduler != None and phase == train_samples:
            #         scheduler.step()
                    
                    
optimiser_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)   
# exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimiser_ft, T_0=10, eta_min=0.00001)    


#  exp_lr_scheduler)

# %tensorboard --logdir runs

def check_accuracy(loader, model):
    model.eval()
    if loader == dataset:
        # model.train()
        print('Checking accuracy on training set')
    else:
        print('Checking accuracy on evaluation set')
        # model.eval()
    num_correct = 0
    num_samples = 0
    #   tells model not to compute gradients
    with torch.no_grad():
        for feature, label in loader:
            feature = feature.to(device)  # move to device
            label = label.to(device)
            scores = model(feature)
            _, preds = scores.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
      
        


if __name__ == '__main__':
    train_model(model, 50, optimiser_ft)
    check_accuracy(dataset, model)
# check_accuracy(val_samples, model)




# %%