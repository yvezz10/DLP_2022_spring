import torch
import torch.nn as nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class resnet_encoder(nn.Module):
    def __init__(self, latent_dim):
        super(resnet_encoder, self).__init__()
        self.latent_dim = latent_dim
        self.model = models.resnet50(pretrained=False)
        self.model.fc = Identity()

    def forward(self, input):
        return self.model(input)


class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class vgg_decoder(nn.Module):
    def __init__(self, dim):
        super(vgg_decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128, 64)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64, 32),
                )
        #128 x 128
        self.upc6 = nn.Sequential(
                vgg_layer(32, 16),
                )
        #256 x 256
        self.upc7 = nn.Sequential(
                #vgg_layer(32, 16),
                nn.ConvTranspose2d(64, 3, 3, 2, 1),
                nn.Tanh()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec= input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(up1) # 8 x 8
        up2 = self.up(d2) # 8 -> 16 
        d3 = self.upc3(up2) # 16 x 16 
        up3 = self.up(d3) # 8 -> 32 
        d4 = self.upc4(up3) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        #d5 = self.upc5(up4) # 64 x 64
        #up5 = self.up(d5)
        #d6 = self.upc6(up5) # 128 x 128
        #up6 = self.up(d6)
        d7 = self.upc7(up4) # 256 x 256
        output = self.up(d7)
        return output