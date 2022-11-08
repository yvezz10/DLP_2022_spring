import json
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from args import parse_args

args = parse_args()

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

def objects2onehot(labels):
    f = open('objects.json')
    datas = json.load(f)

    new_label = []
    for label in labels:
        items = []
        for item in label:
            items.append(datas[item])
        new_label.append(items)
    
    num_classes = 24
    onehot = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        for j in new_label[i]:
            onehot[i,j] = 1.
    
    return onehot

    

class iclevr_data(Dataset):
    def __init__(self, mode = args.mode, transform = default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'new_test'

        self.mode = mode
        self.transform = transform
        self.dirs = []
        self.label = []
        if self.mode == 'train':
            f = open('train.json')

            datas = json.load(f)
            for item in datas.keys():
                self.dirs.append(item)

            for label in datas.values():
                self.label.append(label)

        elif self.mode == 'test':
            f = open('test.json')

            datas = json.load(f)
            for data in datas:
                self.label.append(data)

        elif self.mode == 'new_test':
            f = open('new_test.json')

            datas = json.load(f)
            for data in datas:
                self.label.append(data)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        labels = objects2onehot(self.label)
        label = torch.tensor(labels[index])

        if self.mode == 'train':
            img = Image.open(os.path.join('iclevr',self.dirs[index])).convert('RGB')
            image = self.transform(img)

            return image, label

        else:
            return label
