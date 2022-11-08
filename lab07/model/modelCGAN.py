import torch
import torch.nn as nn
import numpy as np
from args import parse_args

args = parse_args()

img_shape = (3, args.image_shape, args.image_shape)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #self.label_emb = nn.Embedding(args.classes, args.classes)

        self.main = nn.Sequential(
            nn.Linear(args.latent_dim + args.classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            nn.Linear(1024, img_shape[0]*img_shape[1]*img_shape[2]),
            nn.Tanh()
        )

        self._init_weight()

    def forward(self, inputs, labels):

        conditional_inputs = torch.cat([inputs, labels], dim=-1)
        out = self.main(conditional_inputs)
        out = out.reshape(out.size(0), img_shape[0], img_shape[1], img_shape[2])

        return out

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

        #self.label_embedding = nn.Embedding(args.classes, args.classes)

        self.main = nn.Sequential(
            nn.Linear(img_shape[0] * img_shape[1] * img_shape[2] + args.classes, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self._init_weight()

    
    def forward(self, inputs, labels):

        inputs = torch.flatten(inputs, 1)
        #conditional = self.label_embedding(labels)
        #print(conditional.shape)
        #print(inputs.shape)
        conditional_inputs = torch.cat([inputs, labels], dim=-1)
        out = self.main(conditional_inputs).squeeze(1)

        return out

    
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

