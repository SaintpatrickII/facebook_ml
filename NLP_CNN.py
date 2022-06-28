#%%
from turtle import forward
from unicodedata import category
import pandas as pd
import numpy as np
from sqlalchemy import desc
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


products = '/Users/paddy/Desktop/AiCore/facebook_ml/description_for_embedding.pkl'

df =  pd.read_pickle(products)
df.head

class productsPreProcessing(Dataset):
    def __init__(self):
        
        super().__init__()
        df =  pd.read_pickle(products)
        self.descriptions = df['product_description']
        self.categories = df['category']
        self.max_seq_len = 100
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.get_vocab()
        self.descriptions = self.tokenize_descriptions(self.descriptions)
    

    """
    Init:

    We're reading in the dataframe, setting the descriptions and categories as attributes, setting
    the maximum sequence length, getting the tokenizer and vocab, and tokenizing the descriptions
    """


    def get_vocab(self):
       
        def yield_tokens():
            for description in self.descriptions:
                tokens = self.tokenizer(description)
                yield tokens
        token_generator = yield_tokens()

        vocab = build_vocab_from_iterator(token_generator, specials=['<UNK>'])
        print('length of vocab:', len(vocab))
        return vocab


    """
    get_vocab:

    - We create a generator that yields tokens from the descriptions.
    - We use the `build_vocab_from_iterator` function to build the vocabulary from the generator.
    - We return the vocabulary
    :return: A dictionary of words and their corresponding index.
    """


    def tokenize_descriptions(self, descriptions):
        def tokenize_description(description):
            words = self.tokenizer(description)
            words = words[:100]
            pad_length = self.max_seq_len - len(words)
            words.extend(['<UNK>'] * pad_length)
            tokenized_desc = self.vocab(words)
            tokenized_desc = torch.tensor(tokenized_desc)
            return tokenized_desc

        descriptions = descriptions.apply(tokenize_description)
        return descriptions


    """
    Tokenize descriptions:

    We take a dataframe of descriptions, tokenize each description, pad the tokenized descriptions
    to the max sequence length, and return a dataframe of tokenized descriptions
    
    :param descriptions: a pandas series of descriptions
    :return: A list of tokenized descriptions
    """


    def __len__(self):
        return len(self.descriptions)


    '''
    __len__:
    
    overwrites len from Dataset Abstract class
    '''


    def __getitem__(self, idx):
        description = self.descriptions.iloc[idx]
        category = self.categories.iloc[idx]
        return (description, category)

    '''
    __getitem__:
    
    overwrites __getitem__ magic method, required to be able to index items in the dataset
    '''


dataset = productsPreProcessing()
print(dataset[2])



class CNN(torch.nn.Module):
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
            torch.nn.Linear(6272, 13),
            torch.nn.Softmax()
        )


    '''
    CNN initiliser:
    
    We have an embedding layer, which is a matrix of size 26888x100, which is the size of our
    vocabulary. from here a CNN is built using dropout & ReLU to avoid overfitting
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

cnn = CNN()


train_split = 0.7
validation_split = 0.15
batch_size = 32

data_size = len(dataset)
print(f'dataset contains {data_size} Images')

train_size = int(train_split * data_size)
val_size = int(validation_split * data_size)
test_size = data_size - (val_size + train_size)
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_samples = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_samples = DataLoader(test_data, batch_size=batch_size, shuffle=True)

'''
Data Splitter:

takes samples & splits into sample sets, using dataloader to tokenize etc
'''

def train_model(model, epochs):
    writer = SummaryWriter()
    model.train()
    print('training model')
    optimiser = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        for phase in [train_samples, val_samples]:
            if phase == train_samples:
                print('training')
            else:
                print('val')
            for i, (features, labels) in enumerate(phase):
                if phase == 'train':
                    torch.set_grad_enabled(phase)
                num_correct = 0
                num_samples = 0 
                predict = model(features)
                labels = labels
                loss = F.cross_entropy(predict, labels)
                _, preds = predict.max(1)
                num_correct += (preds == labels).sum()
                num_samples += preds.size(0)
                acc = float(num_correct) / num_samples
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                if i % 30 == 29:
                    print(f'Epoch {epoch + 1}/{epochs}')
                    print('-' * 10)
                    if phase == train_samples:
                        writer.add_scalar('Training Loss', loss, epoch)
                        writer.add_scalar(' Training Accuracy', acc, epoch)
                        print('training_loss')
                        print(f'Loss: {loss:.4f} Acc: {acc*100:.1f}%')
                        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                    else:
                        writer.add_scalar('Validation Loss', loss, epoch)
                        writer.add_scalar('Validation Accuracy', acc, epoch)
                        print('val_loss') 
                        # print(batch) # print every 30 mini-batches
                        print(f'Loss: {loss:.4f} Acc: {acc*100:.1f}%')
                        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                        writer.flush()
                

    """
    train model:

    We're training the model, and for each epoch, we're iterating through the training and validation
    samples, and for each mini-batch, we're calculating the loss and accuracy, and then we're
    backpropagating the loss and updating the weights. 

    We're also using a SummaryWriter to write the loss and accuracy to TensorBoard.

    :param model: the model we want to train
    :param epochs: number of times to iterate over the entire dataset
    """

train_model(cnn, 100)


def check_accuracy(loader, model):
    model.eval()
    if loader == train_samples:
        print('Checking accuracy on training set')
    else:
        print('Checking accuracy on evaluation set')
    num_correct = 0
    num_samples = 0
    #   tells model not to compute gradients
    with torch.no_grad():
        for feature, label in loader:
            # feature = feature.to(device)  # move to device
            # label = label.to(device)
            scores = model(feature)
            _, preds = scores.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')


    """
    check accuracy:

    It takes a data loader and a model, and checks the accuracy of the model on the data in the loader
    for both training & validation sets
    :param loader: the data loader for the dataset to check
    :param model: A PyTorch Module giving the model to train
    """


check_accuracy(train_samples, cnn)
check_accuracy(val_samples, cnn)



#%%