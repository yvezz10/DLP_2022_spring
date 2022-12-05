import torch
import torchvision.models as models
import torch.nn as nn
from torch import device, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torchvision.datasets import STL10

from dataset import covid_dataset
from args import parse_args
from ModelAE import resnet_encoder, vgg_decoder
import os 

from sklearn.metrics import classification_report, confusion_matrix
from confusion import plot_confusion_matrix

mse_criterion = nn.MSELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((126, 126)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def denormalize(img):
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0.,0.,0.], std = [1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std = [1., 1., 1.])
    ])

    res = invTrans(img)
    return res

def save_img(img, path, epoch):
    grid = make_grid(img, nrow=8, normalize=8)

    grid = denormalize(grid)
    save_image(grid, format='png', fp= os.path.join(path,"%d_result.png"%epoch))

def train(train_loader, encoder, decoder, optimizer, args, device):

    for e in range(args.epoch):
        encoder.train()
        decoder.train()

        optimizer.zero_grad()
        total_loss = 0

        tbar = tqdm(train_loader)
        for i, (image, label) in enumerate(tbar):

            image = image.to(device)
            emb = encoder(image)
            decode_img = decoder(emb)

            loss = mse_criterion(decode_img, image)
            loss.backward()
            optimizer.step()
            total_loss +=loss

        print("epoch:{:3d}, loss: {:.5f}".format(e, total_loss))
        save_img(decode_img, 'image', e)
        torch.save(encoder.state_dict(), os.path.join(args.model_dir, 'encoder.pt'))
        torch.save(decoder.state_dict(), os.path.join(args.model_dir, 'decoder.pt'))

def main():
    args = parse_args()
    os.makedirs('image',exist_ok=True)
    os.makedirs('weight',exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder = resnet_encoder(args.latent_dim).to(device)
    decoder = vgg_decoder(args.latent_dim).to(device)

    if args.model_dir != '':
        encoder.load_state_dict(torch.load(os.path.join(args.model_dir,'encoder.pt')))
        decoder.load_state_dict(torch.load(os.path.join(args.model_dir,'decoder.pt')))


    #train_data = covid_dataset('train')
    train_data = STL10(root='./STLdata', split='unlabeled', transform=transform, download=False)
    #test_data = covid_dataset('test')

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers= args.num_workers,
                              pin_memory=True)

    #test_loader = DataLoader(test_data,
    #                          batch_size=args.batch_size,
    #                          shuffle=False,
    #                          drop_last=True,
    #                          num_workers= args.num_workers,
    #                          pin_memory=True)
    
    para = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(para, lr = args.lr, betas=(0.1, 0.9))

    train(train_loader, encoder, decoder, optimizer, args, device)

if __name__ =="__main__":
    main()
