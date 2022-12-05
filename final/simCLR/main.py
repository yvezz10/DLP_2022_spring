import torch
import torchvision.models as models
from torchvision.datasets import STL10

from dataset import tranformation
from torch.utils.data import DataLoader

from args import parse_args
from model import SimCLR, Identity
from lossFun import NT_Xent
from coviddataset import covid_dataset

import os
from tqdm import tqdm

trans = tranformation()
torch.backends.cudnn.benchmark = True

def train(args, train_loader, encoder, model, criterion, optimizer, device):

    for e in range(args.epoch):
        loss_epoch = 0
        tbar = tqdm(train_loader)
        for step,  ((x_i, x_j), _) in enumerate(tbar):
            optimizer.zero_grad()

            x_i = x_i.to(device)
            x_j = x_j.to(device)

            h_i = encoder(x_i)
            h_j = encoder(x_j)

            # positive pair, with encoding
            z_i, z_j = model(h_i, h_j)

            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()

            loss_epoch += loss.item()

        loss_epoch /= step
        
        print("epoch:{:4d}, loss:{:.4f}".format(e, loss_epoch))
    torch.save(encoder.state_dict(),'weight/weight.pt')

def main():
    args = parse_args()
    os.makedirs(args.weight_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #train_data = STL10(root='./data', split='unlabeled', transform=trans, download=True)
    train_data = covid_dataset('train')

    trainloader = DataLoader(train_data, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=True,
                            num_workers=args.num_workers)

    encoder = models.resnet50(pretrained=False)
    n_features = encoder.fc.in_features
    encoder.fc = Identity()
    encoder = encoder.to(device)

    model = SimCLR(args.projection_dim, n_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = NT_Xent(args.batch_size, args.temperature)

    train(args, trainloader, encoder, model, criterion, optimizer, device)
    
if __name__ =="__main__":
    main()