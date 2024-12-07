import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class TransformerBlock(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # self attention
        x = self.norm1(x)
        x = x + self.attn(x, x, x)[0]
        
        # feed forward
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x
        
class Transformer(nn.Module):
    
    def __init__(self, dim, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class DownsampleTR(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_heads, mlp_dim=2048, dropout=0.1):
        super().__init__()
        
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.transformer = Transformer(dim=out_channels, depth=depth, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.max_pool(x)
        x = self.relu1(self.conv1(x))
        x = x.permute(0, 2, 1)
        p = self.transformer(x)
        p = p.permute(0, 2, 1)
        x = self.relu2(self.conv2(p))
        return x

class BottleneckTR(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_heads, mlp_dim=2048, dropout=0.1):
        super().__init__()
        
        # self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=25) # TODO: hard-coded at the moment (Maybe remove?)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.transformer = Transformer(dim=out_channels, depth=depth, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        
        # x = self.adaptive_pool(x)
        # print(f"Max pool shape: {x.shape}")
        x = self.relu1(self.conv1(x))
        x = x.permute(0, 2, 1)
        p = self.transformer(x)
        p = p.permute(0, 2, 1)
        x = self.relu2(self.conv2(p))
        
        return x
    
class ProjectionTR(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels * 2, in_channels*4, kernel_size=1),
            nn.BatchNorm1d(in_channels*4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        
        return x
    
class UpsampleTR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # two convolutions with 1x1 kernels
        self.conv_op = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv_op(x)
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
    
##############################################
############## UNETR SETUP ###################
##############################################

class UNETR(nn.Module):
    
    def __init__(self, in_channels, out_channels, img_size=224, feature_size=32, hidden_size=32, mlp_dim=2048, num_heads=8, num_layers=8, dropout=0.1):
        """UNet model with Transformers blocks sprinkled in!

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images or 1 for grayscale images).
            out_channels (int): Number of output channels.
            img_size (int, optional): Input image size (assumes square images). Defaults to 224.
            feature_size (int, optional): Number of feature maps in the first layer of the UNet encoder. Defaults to 32.
            hidden_size (int, optional): Dimensionality of the Transformer embedding space. Defaults to 768.
            mlp_dim (int, optional): Dimensionality of the MLP (feedforward) layer in the Transformer. Defaults to 2048.
            num_heads (int, optional): Number of attention heads in the Transformer. Defaults to 12.
            num_layers (int, optional): Number of Transformer layers. Defaults to 12.
            dropout (float, optional): Dropout rate for Transformer layers. Defaults to 0.1.
        """
        super().__init__()
        
        self.patch_size = 16 # TODO: hard-coded to match pooling requirements
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** 2
        self.feature_size = feature_size
        self.transformer_args = {
            "depth": num_layers,
            "num_heads": num_heads,
            "mlp_dim": mlp_dim,
            "dropout": dropout
        }
        
        # positional Embeddings
        self.positional_embeddings = nn.Parameter(torch.zeros(1, hidden_size, self.num_patches))

        # patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        
        # encoder
        self.encoder1 = DoubleConv1D(hidden_size, feature_size)
        self.encoder2 = DownsampleTR(
            in_channels=feature_size, out_channels=feature_size*2, **self.transformer_args
        )
        self.encoder3 = DownsampleTR(
            in_channels=feature_size*2, out_channels=feature_size*4, **self.transformer_args
        )
        
        # bottleneck
        self.bottleneck = BottleneckTR(
            in_channels=feature_size*4, out_channels=feature_size*8, **self.transformer_args
        )
        
        # projection for upsampling
        self.projection = ProjectionTR(in_channels=feature_size*8)
        
        # upsample
        self.decoder1 = UpsampleTR(in_channels=feature_size*32, out_channels=feature_size*16)
        self.decoder2 = UpsampleTR(in_channels=feature_size*16, out_channels=feature_size*8)
        self.decoder3 = UpsampleTR(in_channels=feature_size*8, out_channels=feature_size*4)
        self.decoder4 = UpsampleTR(in_channels=feature_size*4, out_channels=feature_size*2)
        
        # output layer
        self.out = nn.ConvTranspose2d(
            in_channels=feature_size*2, 
            out_channels=out_channels, 
            kernel_size=4, 
            stride=2,
            padding=1
        )
        
    def forward(self, x):
        
        ### patch embedding ###
        x = self.patch_embedding(x)
        x = x.flatten(2)
        # print(f"Patch embedding shape: {x.shape}")
        
        ### add positional embeddings ###
        x = x + self.positional_embeddings
        # print(f"Positional embeddings shape: {x.shape}")
        
        ### encoder ###
        enc1 = self.encoder1(x)
        # print(f"Encoder 1 shape: {enc1.shape}")
        enc2 = self.encoder2(enc1)
        # print(f"Encoder 2 shape: {enc2.shape}")
        enc3 = self.encoder3(enc2)
        # print(f"Encoder 3 shape: {enc3.shape}")
        
        ### bottleneck ###
        bottle = self.bottleneck(enc3)
        # print(f"Bottleneck shape: {bottle.shape}")
        
        ### projection ###
        proj = self.projection(bottle)
        # print(f"Projection shape: {proj.shape}")
        seq_len = proj.size(2)
        spatial_dim = int(seq_len ** 0.5)
        proj = proj.view(proj.size(0), proj.size(1), spatial_dim, spatial_dim)
        # print(f"Projection reshaped shape: {proj.shape}")
        
        ### upsample ###
        dec1 = self.decoder1(proj)
        # print(f"Decoder 1 shape: {dec1.shape}")
        dec2 = self.decoder2(dec1)
        # print(f"Decoder 2 shape: {dec2.shape}")
        dec3 = self.decoder3(dec2)
        # print(f"Decoder 3 shape: {dec3.shape}")
        dec4 = self.decoder4(dec3)
        # print(f"Decoder 4 shape: {dec4.shape}")
        
        ### output layer ###
        out = self.out(dec4)
        # print(f"Output shape: {out.shape}")
    
        return out