import torch
import torch.nn as nn
import torch.nn.functional as F
from refiner_model import _RefinerG

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class _netG1(nn.Module):
    def __init__(self, nz, ngf, batch_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf//2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf//4, ngf//8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf//8, 100, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        if input.dim() == 2:
            input = input.view(input.size(0), input.size(1), 1, 1)
        return self.main(input)

class _netD1(nn.Module):
    def __init__(self, ndf, nz=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nz, ndf//8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//8, ndf//16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf//16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//16, 1, 1, 1, 0, bias=False)
        )
        self.apply(weights_init)

    def forward(self, input):
        if input.dim() == 2:
            input = input.view(input.size(0), -1, 1, 1)
        elif input.dim() == 4:
            if input.size(2) != 1 or input.size(3) != 1:
                input = input.mean(dim=[2,3], keepdim=True)
        return self.main(input).view(-1, 1)

class _netG2(nn.Module):
    def __init__(self, nz, ngf, nc, batch_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf//2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf//4, ngf//8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf//8, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        if output.size(2) != 64 or output.size(3) != 64:
            output = F.interpolate(output, size=(64, 64), mode='bilinear', align_corners=False)
        return output

class _netRS(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf//4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//4, ndf//8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf//8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//8, 100, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)

class _RefinerG(nn.Module):
    def __init__(self, nc, ngf):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ngf//4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf//4, ngf//8, 4, 2, 1),
            nn.BatchNorm2d(ngf//8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf//8, ngf//4, 4, 2, 1),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf//4, nc, 4, 2, 1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.encoder(input)
        return self.decoder(x)

class _RefinerD(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf//4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//4, ndf//8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf//8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf//8, 1, 4, 1, 0, bias=False)
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input).view(-1, 1)