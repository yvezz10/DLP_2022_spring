import torch
import torch.nn as nn
import numpy as np
from args import parse_args

args = parse_args()

img_shape = (3, args.image_shape, args.image_shape)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb =  nn.Sequential(
            nn.Linear(args.classes, args.emb_dim),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.emb_dim + args.latent_dim, 512, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            *self.make_block(512, 256),
            *self.make_block(256, 128),
            *self.make_block(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.con_emb = nn.Sequential(
            nn.Linear(args.emb_dim + args.latent_dim, args.emb_dim + args.latent_dim),
            nn.ReLU()
        )

        self._init_weight()

    def forward(self, inputs, labels):

        z = inputs
        c = self.label_emb(labels)

        conditional_inputs = self.con_emb(torch.cat([z, c], dim=1))
        conditional_inputs = conditional_inputs.view(-1, args.emb_dim + args.latent_dim, 1, 1)
        out = self.main(conditional_inputs)
        
        return out

    def make_block(self, input, output):
        block = [nn.ConvTranspose2d(input, output, kernel_size=4, stride=2, padding=1)]
        block.append(nn.BatchNorm2d(output))
        block.append(nn.LeakyReLU(negative_slope=0.2, inplace=True),)
        return block

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(args.classes, img_shape[2]*img_shape[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.main = nn.Sequential(
            *self.make_block(4, 64),
            *self.make_block(64,128),
            *self.make_block(128,256),
            *self.make_block(256,512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
            #nn.Sigmoid()
        )

        self._init_weight()

    
    def forward(self, inputs, labels):

        c = self.embedding(labels).view(-1, 1, img_shape[2], img_shape[1])
        x = torch.cat((inputs, c), dim=1)
        out = self.main(x).view(-1)

        return out

    def make_block(self, input, output):
        block = [nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1)]
        block.append(nn.BatchNorm2d(output))
        block.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        block.append(nn.Dropout(0.5))
        return block
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

