from dataset import iclevr_data
from args import parse_args
from evaluator import evaluation_model

from modelWGANGP import Generator, Discriminator
import os
import copy
import numpy as np
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import grad as torch_grad

torch.backends.cudnn.benchmark = True

def getCheckPoint(args):
    import pandas as pd
    
    if os.path.exists(os.path.join(args.log_dir,'log.csv')):
        df = pd.read_csv(os.path.join(args.log_dir,'log.csv'))
        try:
            last_epoch = df.iloc[-1]['epoch']+1
        except:
            last_epoch = args.last_e

    else:
        last_epoch = args.last_e

    return int(last_epoch)


class logCsv():
    def __init__(self, args):
        self.args = args

        os.makedirs(self.args.log_dir, exist_ok=True)

    def createFile(self):
        #if os.path.exists(os.path.join(self.args.log_dir,'log.csv')):
            #raise RuntimeError("Remove the previous log file if trying to train from scratch")

        with open(os.path.join(self.args.log_dir,'log.csv'), 'w') as f:
            writer = csv.writer(f)
            header = ['epoch', 'generator loss', 'discriminator loss', 'train score', 'test score', 'new test score']
            writer.writerow(header)

    def logData(self, epoch, g_loss, d_loss, train_score, test_score, new_test_score):
        with open(os.path.join(self.args.log_dir,'log.csv'), 'a+') as f:
            writer = csv.writer(f)
            log = [epoch, g_loss.detach().cpu().numpy(), d_loss.detach().cpu().numpy(), train_score, test_score, new_test_score]
            writer.writerow(log)

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

def get_gradient_penalty(discriminator, real, fake, label, args, device):

    lambda_gp = 10
    alpha = torch.rand(args.batch_size, 1, 1, 1).to(device)
    alpha = alpha.expand_as(real)
    

    interpolated = alpha*real.data + (1-alpha)*fake.data

    interpolated = Variable(interpolated, requires_grad = True).to(device)
    
    prob_interpolated = discriminator(interpolated, label)

    gradients = torch_grad(outputs=prob_interpolated,
                           inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True,
                           retain_graph=True,
                           only_inputs=True
                          )[0]
    gradients = gradients.view(args.batch_size, -1)
    gradients_norm = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return lambda_gp * gradients_norm



def train(dataloader, testloader, new_test_loader, generator, discriminator, args, device, logger, lastCk):

    #optimizer_gen = optim.RMSprop(generator.parameters(), lr = args.lr)
    #optimizer_dis = optim.RMSprop(discriminator.parameters(), lr = args.lr)

    optimizer_gen = optim.Adam(generator.parameters(), lr = args.lr, betas=(0, 0.9))
    optimizer_dis = optim.Adam(discriminator.parameters(), lr = args.lr, betas=(0, 0.9))

    max_acc = 1.2
    evaluator = evaluation_model()

    best_generator = None
    best_discriminator = None

    for e in range(args.epochs):

        generator.train()
        discriminator.train()

        sum_d_loss = 0
        sum_g_loss = 0
        accuracy = []

        tbar = tqdm(dataloader)

        for i, (img, label) in enumerate(tbar):

            img, label = Variable(img.type(torch.FloatTensor)).to(device), Variable(label.type(torch.FloatTensor)).to(device)

            noise = torch.randn([args.batch_size, args.latent_dim]).to(device)

            ######train discriminator######
            d_iter = 1
            for i in range(d_iter):
                discriminator.zero_grad()

                #train with real data
                real_output = discriminator(img, label)

                #train with fake data
                fake_img = generator(noise, label)
                fake_output = discriminator(fake_img, label)

                gradient_penalty = get_gradient_penalty(discriminator, img, fake_img, label, args, device)

                d_loss = -torch.mean(real_output) + torch.mean(fake_output) + gradient_penalty
                d_loss.backward()
                optimizer_dis.step()

                sum_d_loss += (d_loss/d_iter)


                #for p in discriminator.parameters():
                    #p.data.clamp_(-args.clip_value, args.clip_value)

            ######train generator######
            g_iter = 4
            for i in range(g_iter):
                generator.zero_grad()
                noise = torch.randn([args.batch_size, args.latent_dim]).to(device)
                generate_img = generator(noise, label)

                predict = discriminator(generate_img, label)
                loss_g = -torch.mean(predict)

                loss_g.backward()
                optimizer_gen.step()
                sum_g_loss += (loss_g/g_iter)

                acc = evaluator.eval(generate_img, label)

                accuracy.append(acc)

        if (e+lastCk)%10 == 0 or e == (args.epochs -1):
            save_img(generate_img,os.path.join(args.img_dir, 'train'), e+lastCk)

        sum_d_loss = sum_d_loss/args.batch_size
        sum_g_loss = sum_g_loss/args.batch_size
        
        score = np.mean(accuracy)
        
        ######test data score #####
        generator.eval()
        with torch.no_grad():
            for _, label in enumerate(testloader):
                noise = torch.randn([32, args.latent_dim]).to(device)
                label = Variable(label.type(torch.FloatTensor)).to(device)
                generate_img = generator(noise, label)
                test_score = evaluator.eval(generate_img, label)

            for _, label in enumerate(new_test_loader):
                noise = torch.randn([32, args.latent_dim]).to(device)
                label = Variable(label.type(torch.FloatTensor)).to(device)
                generate_img = generator(noise, label)
                new_test_score = evaluator.eval(generate_img, label)

        #test_acc = test_score + new_test_score
        test_acc = float('{:.1f}'.format(test_score))+float('{:.1f}'.format(new_test_score))
        if test_acc> max_acc:
            max_acc = test_score
            best_generator = copy.deepcopy(generator)
            best_discriminator = copy.deepcopy(discriminator)

        logger.logData(e+lastCk, sum_g_loss, sum_d_loss, score, test_score, new_test_score)

        print("epoch:{:4d}, generator loss:{:4.5f}, discriminator loss:{:4.5f}, train socre:{:.3f}, test socre:{:.3f}, new test score:{:.3f}"\
            .format(e+lastCk, sum_g_loss, sum_d_loss, score, test_score, new_test_score))

        if (e+1)%25 == 0:
            torch.save(generator.state_dict(), os.path.join(args.weight_dir, 'generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.weight_dir, 'discriminator.pth'))
            if best_generator is not None:
                torch.save(best_generator.state_dict(), os.path.join(args.weight_dir, 'best_generator.pth'))
                torch.save(best_discriminator.state_dict(), os.path.join(args.weight_dir, 'best_discriminator.pth'))

def eval(dataloader, generator, new_test_loader, args, device):

    generator.eval()

    evaluator = evaluation_model()
    test_accuracy  = []
    new_test_accuracy = []
    with torch.no_grad():
        tbar = tqdm(dataloader)
        for i, label in enumerate(tbar):
            noise = torch.randn([32, args.latent_dim]).to(device)
            label = Variable(label.type(torch.FloatTensor)).to(device)
            generate_img_test = generator(noise, label)
            acc = evaluator.eval(generate_img_test, label)
            test_accuracy.append(acc)

        tbar = tqdm(new_test_loader)
        for i, label in enumerate(tbar):
            noise = torch.randn([32, args.latent_dim]).to(device)
            label = Variable(label.type(torch.FloatTensor)).to(device)
            generate_img = generator(noise, label)
            acc = evaluator.eval(generate_img, label)
            new_test_accuracy.append(acc)

    test_score = np.mean(test_accuracy)
    new_test_score = np.mean(new_test_accuracy)
    save_img(generate_img_test,os.path.join(args.img_dir, 'test'), 0)
    save_img(generate_img,os.path.join(args.img_dir, 'test_new'), 0)
    print("test score:{:.2f}, new test score:{:.2f}".format(test_score, new_test_score))



def main():
    args = parse_args()

    logger = logCsv(args)

    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs(os.path.join(args.img_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.img_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.img_dir, 'test_new'), exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("using device", device)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    train_data = iclevr_data(mode='train')
    test_data = iclevr_data(mode='test')
    new_test_data = iclevr_data(mode='new_test')

    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last= True,
                            pin_memory=True)

    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=32,
                            shuffle=False,
                            pin_memory=True)

    new_test_loader = DataLoader(new_test_data,
                            num_workers=args.num_workers,
                            batch_size=32,
                            shuffle=False,
                            pin_memory=True)

    if args.model_dir !='':
        generator.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_generator.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_discriminator.pth')))
        last_epoch = getCheckPoint(args)
    else:
        logger.createFile()
        last_epoch = 0

    train(train_loader, test_loader, new_test_loader, generator, discriminator, args, device, logger, last_epoch)
    #eval(test_loader, generator, new_test_loader, args, device)

if __name__ =='__main__':
    main()


