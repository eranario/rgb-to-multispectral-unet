import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # two convolutions with 3x3 kernel
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        
        return self.conv_op(x)
    
class Downsample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        down = self.double_conv(x)
        p = self.max_pool(down)
        
        return down, p
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.double_conv_half = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x1, x2], dim=1)
            x = self.double_conv(x)
        else:
            x = x1  # No skip connection
            x = self.double_conv_half(x)
        return x
    
##############################################
############### UNET SETUP ###################
##############################################
    
class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # decoder
        self.down_conv_1 = Downsample(in_channels, out_channels=64)
        self.down_conv_2 = Downsample(in_channels=64, out_channels=128)
        self.down_conv_3 = Downsample(in_channels=128, out_channels=256)
        self.down_conv_4 = Downsample(in_channels=256, out_channels=512)
        
        # bottleneck
        self.bottle_neck = DoubleConv(in_channels=512, out_channels=1024)
        
        # encoder
        self.up_conv_1 = Upsample(in_channels=1024, out_channels=512)
        self.up_conv_2 = Upsample(in_channels=512, out_channels=256)
        self.up_conv_3 = Upsample(in_channels=256, out_channels=128)
        self.up_conv_4 = Upsample(in_channels=128, out_channels=64)
        
        # output layer
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        
        # decoder
        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)
        
        # bottleneck
        b = self.bottle_neck(p4)
        
        # encoder
        up_1 = self.up_conv_1(b, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)
        
        # output layer
        out = self.out(up_4)
        return out
    
##############################################
################ VAE SETUP ###################
##############################################

class VAE(nn.Module):
    
    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        
        # encoder
        self.down_conv_1 = Downsample(in_channels, out_channels=64)
        self.down_conv_2 = Downsample(in_channels=64, out_channels=128)
        self.down_conv_3 = Downsample(in_channels=128, out_channels=256)
        self.down_conv_4 = Downsample(in_channels=256, out_channels=512)
        
        # bottleneck
        self.bottleneck_mu = nn.Linear(512*14*14, latent_dim) # mean
        self.bottleneck_logvar = nn.Linear(512*14*14, latent_dim) # log variance
        
        # decoder
        self.decoder_input = nn.Linear(latent_dim, 512*14*14)
        self.up_conv_1 = Upsample(in_channels=512, out_channels=256)
        self.up_conv_2 = Upsample(in_channels=256, out_channels=128)
        self.up_conv_3 = Upsample(in_channels=128, out_channels=64)
        self.up_conv_4 = Upsample(in_channels=64, out_channels=32)
        
        # output layer
        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)
        
    def reparameterize(self, mu, logvar):
        """To sample from N(mu, var) using N(0, 1)"""
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        """Encode input into latent space"""
        _, p1 = self.down_conv_1(x)
        _, p2 = self.down_conv_2(p1)
        _, p3 = self.down_conv_3(p2)
        _, p4 = self.down_conv_4(p3)

        # flatten
        p4 = p4.view(p4.size(0), -1)
        mu = self.bottleneck_mu(p4)
        logvar = self.bottleneck_logvar(p4)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent space into output"""
        x = self.decoder_input(z)
        x = x.view(x.size(0), 512, 14, 14)
        
        up_1 = self.up_conv_1(x, None)
        up_2 = self.up_conv_2(up_1, None)
        up_3 = self.up_conv_3(up_2, None)
        up_4 = self.up_conv_4(up_3, None)
        
        out = self.out(up_4)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
##############################################
################ DCGAN SETUP #################
##############################################

class Generator(nn.Module):
    def __init__(self, noise_dim, out_channels, img_size=224):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim, 512 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):

        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)

        img = self.conv_blocks(out)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, in_channels, img_size=224):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        ds_size = img_size // 2 ** 5
        self.fc = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out