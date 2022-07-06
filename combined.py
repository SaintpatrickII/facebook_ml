#%%
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io


products_df = '/Users/paddy/Desktop/AiCore/facebook_ml/df_from_demo.csv'
image_folder = '/Users/paddy/Desktop/AiCore/facebook_ml/images_raw'

class ImageTextDataloader(torch.utils.data.Dataset):

    def __init__(self, Image_dir, csv_file, transform: transforms = None, labels_level : int=0, max_desc_len = 50):
        self.products = pd.read_csv(csv_file, lineterminator='\n')
        self.root_dir = Image_dir
        self.transform = transform


        self.max_desc_len = max_desc_len
        self.products['category'] = self.products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.descriptions = self.products['product_description']
        self.image_id = self.products['id']
        self.labels = self.products['category'].to_list()
        self.num_classes = len(set(self.labels))


        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        if transform == None:
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(128)
                ])


        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.get_vocab()
        self.descriptions = self.tokenize_descriptions(self.descriptions)

        assert len(self.descriptions) == len(self.labels) == len(self.image_id)
    

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
            words = words[:50]
            pad_length = self.max_desc_len - len(words)
            words.extend(['<UNK>'] * pad_length)
            tokenized_desc = self.vocab(words)
            tokenized_desc = torch.tensor(tokenized_desc)
            return tokenized_desc

        descriptions = descriptions.apply(tokenize_description)
        return descriptions


    def __len__(self):
        return len(self.products)


    def __getitem__(self, index):
        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        print(self.products.iloc[index, 0])
        image = os.path.join(self.root_dir, (self.products.iloc[index, 0].astype(str)))
        image = torch.tensor(image).float()
        # print(image)
        image = io.imread(f'{image}.jpg')
        image = self.transform(image)
        description = self.descriptions[index]
        # encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        # encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        # with torch.no_grad():
        #     description = self.model(**encoded).last_hidden_state.swapaxes(1,2)

        # description = description.squeeze(0)

        return image, description, label




    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()


if __name__ == '__main__':
    dataset = ImageTextDataloader(Image_dir='/Users/paddy/Desktop/AiCore/facebook_ml/Images', csv_file='/Users/paddy/Desktop/AiCore/facebook_ml/df_from_demo.csv')
    # print(dataset[2500])
    # print(dataset.decoder[int(dataset[2500][2])])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,
                                             shuffle=True, num_workers=1)
    for i, (image, description, labels) in enumerate(dataloader):
        print(image)
        print(labels)
        print(description.size())
        print(image.size())
        if i == 0:
            break

#%%