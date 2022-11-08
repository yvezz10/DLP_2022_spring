import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from args import parse_args

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'

        if mode == 'train':
            self.data_dir = os.path.join(args.data_root,'train')
            self.ordered = False
        elif mode == 'validate':
            self.data_dir = os.path.join(args.data_root,'validate')
            self.ordered = True
        else:
            self.data_dir = os.path.join(args.data_root,'test')
            self.ordered = True
        
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir,d1,d2))

        self.seed_is_set = False
        self.d = 0
        self.transform = transform
        self.n_eval = args.n_eval
        self.mode = mode
        self.batch_size = args.batch_size

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        self.d_now = d

    def __len__(self):
        return len(self.dirs)

    def get_seq(self):
        image_seq = []
        for i in range(self.n_eval):
            fname = '%s/%d.png' % (self.d_now, i)
            im = Image.open(fname)
            img = np.array(im).reshape(1, 64, 64, 3)/255.
            image_seq.append(img)

        image_seq = np.concatenate(image_seq, axis=0)
        image_seq = torch.from_numpy(image_seq).float()
        image_seq = image_seq.permute(0,3,1,2)
        
        return image_seq 
    
    def get_csv(self):

        action_list = []
        with open(('%s/actions.csv'%(self.d_now)), newline='') as f:
            reader = csv.reader(f)
            act_list = list(reader)
            action_list.append(act_list[:self.n_eval])

        position_list = []
        with open(('%s/endeffector_positions.csv'%(self.d_now)), newline='') as f:
            reader = csv.reader(f)
            pos_list = list(reader)
            position_list.append(pos_list[:self.n_eval])

        action_list = np.concatenate(action_list, axis =0)
        position_list = np.concatenate(position_list, axis=0)

        condition = np.concatenate([action_list, position_list], axis=1)
        condition = condition.astype(float)

        ##one hot encoding
        
        onehot = condition[:,2:4]
        onehot = onehot.astype(int)
        num_classess = 5
        onehot = np.eye(num_classess)[onehot]
        onehot = onehot.astype(float)
        onehot = np.reshape(onehot, (self.n_eval, 2*num_classess))
        condition = np.delete(condition, [2,3], axis=1)
        condition =np.concatenate([condition, onehot], axis=1)

        return torch.tensor(condition).float()

    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq()
        cond =  self.get_csv()
        return seq, cond

if __name__ == "__main__":
    data = bair_robot_pushing_dataset(args=parse_args(), mode='validate')
    test = data.__getitem__(1)


