from unicodedata import category
import pandas as pd
import numpy as np
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
        self.max_seq_len = 128
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
        return vocab


    def tokenize_descriptions(self, descriptions):
        def tokenize_description(description):
            words = self.tokenizer(description)
            words = words[:128]
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