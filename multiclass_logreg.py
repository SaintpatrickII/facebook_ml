import json
import glob
import pandas as pd
import numpy as np
import sklearn
from numpy import asarray
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch import flatten
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/image_data.csv'
images = '/Users/paddy/Desktop/AiCore/facebook_ml/images_raw'
images_corrected = '/Users/paddy/Desktop/AiCore/facebook_ml/Images'

df = pd.read_csv(data)

class image_setup:
    def __init__(self, df: pd):
        self.df = df
        self.image_list_fixed = glob.glob(f'{images_corrected}/*.jpg')
        self.image_id_list = []
        self.class_list = []
        self.flattened = []
        self.list_of_tuples = []
        self.df_cleaned(df)
        self.classification_names_list(df)
        self.image_correct_prop()
        self.image_to_tensor()
        self.training_tuples()

    
    def df_cleaned(self, df: pd):
        self.df = self.df.drop(['Unnamed: 0', 'bucket_link', 'image_ref', 'create_time'], axis=1)
        self.df = self.df.drop(['uuid', 'price', 'id.1'], axis=1)
        column_indicies = [1,2,3,4,5,6,7,8,9,10,11,12]
        classification_columns = ['0','1','2','3','4','5','6','7','8','9','10','11']
        old_names = self.df.columns[column_indicies]
        self.df.rename(columns=dict(zip(old_names, classification_columns)),inplace=True)
        self.df = self.df.drop(['id'], axis=1)


    def classification_names_list(self, df: pd):
        self.df['new'] = self.df.apply(lambda x: x.index[x == 1].tolist(), axis=1)
        flat_list = []
        class_list = self.df['new'].tolist()
        for sublist in class_list:
            for item in sublist:
                flat_list.append(item)
            self.class_list = list(map(int, flat_list))


    def image_correct_prop(self):
        resized_img = 128
        for n, img in enumerate(image_list[:50], 1):
            image_ite = Image.open(img)
            background = Image.new(mode='RGB', size=(resized_img, resized_img)) 
            original_img = image_ite.size
            max_dim = max(image_ite.size)
            ratio = resized_img / max_dim
            img_corr_ratio =(int(original_img[0] * ratio), int(original_img[1] * ratio))
            img_corr = image_ite.resize(img_corr_ratio)
            # print(img_corr)
            background.paste(img_corr, (((resized_img - img_corr_ratio[0]) // 2), ((resized_img - img_corr_ratio[1]) // 2)))
            # print(background)
            background.save(f'{images_corrected}/{img[50:]}')

    
    def image_to_tensor(self):
        for i, img in enumerate(self.image_list_fixed[:50]):
            image = Image.open(img)
            t = ToTensor()
            tensor = t(image)
            flattened = torch.flatten(tensor)
            # print(self.flattened)
            self.flattened.append(flattened)

        return self.flattened

    def training_tuples(self):
        for i in range(6266):
                # print(image_array[i])
                # print(image_array[i])
                ready_tuple = (self.flattened[i], self.class_list[i])
                self.list_of_tuples.append(ready_tuple)
        return self.list_of_tuples
        


            
    
    pass
image_setup(df)