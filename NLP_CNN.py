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



products = '/Users/paddy/Desktop/AiCore/facebook_ml/nlp_fin.pkl'

df =  pd.read_pickle(products)
df.head

#%%
# %%
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



    def get_vocab(self):
        def yield_tokens():
            for description in self.descriptions:
                tokens = self.tokenizer(description)
                yield tokens
        token_generator = yield_tokens()

        vocab = build_vocab_from_iterator(token_generator, specials=['<UNK>'])
        print('length of vocab:', len(vocab))
        return vocab


    def tokenize_descriptions(self, descriptions):
        def tokenize_description(description):
            words = self.tokenizer(description)
            words = words[:100]
            pad_length = self.max_seq_len - len(words)
            words.extend(['<UNK>']*pad_length)
            tokenized_desc = self.vocab(words)
            tokenized_desc = torch.tensor(tokenized_desc)
            return tokenized_desc

        descriptions = descriptions.apply(tokenize_description)
        return descriptions


    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions.iloc[idx]
        category = self.categories.iloc[idx]
        return (description, category)



dataset = productsPreProcessing()

print(dataset[2])
#%%



class CNN(torch.nn.Module):
    def __init__(self, pretrained_weights=None):
        super().__init__()
        no_words = 26888
        embedding_size = 16
        self.embedding = torch.nn.Embedding(no_words, embedding_size)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(embedding_size, 32, 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(98, 128),
            torch.nn.Softmax()
        )


    def forward(self, X):
        print(X.shape)
        X = self.embedding(X)
        X = X.transpose(2, 1)
        print(X.shape)
        print(X)
        return self.layers(X)

cnn = CNN()


example = dataset[1]
description, category = example
prediction = cnn(description.unsqueeze(0))
print(prediction)
print(category)

#%%
