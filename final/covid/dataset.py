import pandas as pd
from PIL import Image
import os
from args import parse_args
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import torch

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((254, 254)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip()
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((254, 254)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_image.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_image.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class covid_dataset(Dataset):
    def __init__(self, mode):
        assert mode == 'train' or mode == 'test'
        
        self.img_name, self.label = getData(mode)
        self.mode = mode
    
    def __len__(self):
        return(len(self.img_name))

    def __getitem__(self, index):
        
        label = torch.tensor(self.label[index])
        img = Image.open(os.path.join('dataset','images', self.img_name[index]+'.png'))
        back = Image.new(mode = 'RGB', size=(299,299))
        msk = Image.open(os.path.join('dataset','masks', self.img_name[index]+'.png')).resize((299, 299)).convert('L')
        image = Image.composite(img, back, msk)
        transform = train_transform if self.mode == 'train' else test_transform

        image = transform(image)

        return image, label

    def getWeight(self):
        count = np.bincount(self.label)
        maxClass = np.nanmax(count)
        self.data_weight = maxClass/count
        
        return torch.tensor(self.data_weight).float()



