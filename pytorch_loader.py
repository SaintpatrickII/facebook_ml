import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from skimage import io


class ImagesLoader(Dataset):

    def __init__(self, json_file, root_dir, transform=None):
        self.read_json = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.read_json)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.read_json.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.read_json.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return(image, y_label)