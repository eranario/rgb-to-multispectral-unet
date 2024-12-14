import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.transformer_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu"
        )
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=1)

    def forward(self, x):
        # Flatten spatial dimensions to sequence (B, C, H, W) -> (B, C, HW) -> (HW, B, C)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Reshape back to spatial dimensions (HW, B, C) -> (B, C, H, W)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x
    
class TransformerBlockSpectral(nn.Module):
    def __init__(self, embed_dim, signal_dim, num_heads, ff_dim, patch_size, dropout=0.3):
        super(TransformerBlockSpectral, self).__init__()
                
        # transformer layers
        self.patch_embed = PatchEmbedding(in_channels=embed_dim, embed_dim=embed_dim, patch_size=patch_size)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # projection for signal
        self.proj = nn.Linear(signal_dim, embed_dim)
        
        # positional embeddings
        self.pos_embed = None
        
    def forward(self, x, signal=None, mask=None):
        
        b, c, h, w = x.shape
        
        # apply patch embedding
        x, (ph, pw) = self.patch_embed(x)
        
        # create positional embeddings
        if self.pos_embed is None or self.pos_embed.size(1) != ph * pw:
            self.pos_embed = nn.Parameter(torch.zeros(1, ph * pw, x.size(2), device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
        x = x + self.pos_embed
        
        # create patch mask
        if mask is not None:
            patch_mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=(ph, pw),
                mode="nearest"
            ).squeeze(1)
            patch_mask = (patch_mask > 0.5).to(torch.bool)
            patch_mask = patch_mask.view(b, -1) # flatten to (b, num_patches)
                
        # self attention
        x = x.transpose(0, 1)
        attn_output, self_weights = self.self_attn(
            x, x, x, key_padding_mask=~patch_mask if patch_mask is not None else None
        )
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # cross attention
        cross_weights = None
        if signal is not None:
            signal = self.proj(signal)
            signal = signal.transpose(0, 1)
            
            attn_output, cross_weights = self.cross_attn(
                x, signal, signal
            ) # TODO: try V as image
            x = x + self.dropout(attn_output)
            x = self.norm2(x)
            
        # feed forward
        x = x.permute(1, 0, 2)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        # reshape back to b, c, h, w
        x = x.permute(0, 2, 1).view(b, c, ph, pw)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) # TODO: maybe this makes things worst
        
        return x, self_weights, cross_weights
    
class SpectralEmbedding(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_tokens):
        super(SpectralEmbedding, self).__init__()

        self.embed_dim = embed_dim
        
        # ensure input_dim is divisible by num_tokens
        if input_dim % num_tokens != 0:
            # find the closest number of tokens that ensures divisibility
            num_tokens = self.find_closest_divisible(input_dim, num_tokens)
            print(f"Adjusted num_tokens to {num_tokens} to make input_dim divisible.")
        
        self.num_tokens = num_tokens
        self.token_size = input_dim // self.num_tokens
        
        self.proj = nn.Linear(self.token_size, embed_dim) # TODO: Maybe add non-linearity
        
    @staticmethod
    def find_closest_divisible(input_dim, num_tokens):
        """
        Find the closest number to num_tokens that ensures divisibility of input_dim.
        """
        upper = num_tokens
        while upper < input_dim:
            if input_dim % upper == 0:
                return upper
            upper += 1
        return num_tokens  # fallback (shouldn't be reached)
        
    def forward(self, x):
        
        # split into tokens
        bs, _, _ = x.shape
        x = x.view(bs, self.num_tokens, self.token_size) # bs, num_tokens, token_size

        # apply linear transformation
        x = self.proj(x) # bs, num_tokens, embed_dim
    
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = None
        
    def forward(self, x):
        _, _, h, w = x.shape
        
        patch_size = self.patch_size
        while h % patch_size != 0 or w % patch_size != 0:
            patch_size -= 1
            if patch_size < 1:
                raise ValueError("Invalid patch size")

        if self.proj is None or self.proj.kernel_size != (patch_size, patch_size):
            self.proj = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ).to(x.device)
            
        x = self.proj(x) # b, embed_dim, h // patch_size, w // patch_size

        # reshape to sequence format
        _, _, ph, pw = x.shape
        x = x.flatten(2).transpose(1, 2) # b, num_patches, embed_dim
        
        return x, (ph, pw)
    
##############################################
############## UNET with TR ##################
##############################################

class UNeTransformed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNeTransformed, self).__init__()
        
        # Decoder
        self.down_conv_1 = Downsample(in_channels, out_channels=64)
        self.trans_1 = TransformerBlock(embed_dim=64, num_heads=1, ff_dim=256)
        self.down_conv_2 = Downsample(in_channels=64, out_channels=128)
        self.trans_2 = TransformerBlock(embed_dim=128, num_heads=1, ff_dim=512)
        self.down_conv_3 = Downsample(in_channels=128, out_channels=256)
        self.trans_3 = TransformerBlock(embed_dim=256, num_heads=2, ff_dim=1024)
        self.down_conv_4 = Downsample(in_channels=256, out_channels=512)
        self.trans_4 = TransformerBlock(embed_dim=512, num_heads=2, ff_dim=2048)
        
        # Bottleneck
        self.bottle_neck = DoubleConv(in_channels=512, out_channels=1024)
        
        # Encoder
        self.up_conv_1 = Upsample(in_channels=1024, out_channels=512)
        self.up_conv_2 = Upsample(in_channels=512, out_channels=256)
        self.up_conv_3 = Upsample(in_channels=256, out_channels=128)
        self.up_conv_4 = Upsample(in_channels=128, out_channels=64)
        
        # Output layer
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        # Decoder
        down_1, p1 = self.down_conv_1(x)
        trans_1 = self.trans_1(down_1)
        down_2, p2 = self.down_conv_2(p1)
        trans_2 = self.trans_2(down_2)
        down_3, p3 = self.down_conv_3(p2)
        trans_3 = self.trans_3(down_3)
        down_4, p4 = self.down_conv_4(p3)
        trans_4 = self.trans_4(down_4)
        
        # Bottleneck
        b = self.bottle_neck(p4)
        
        # Encoder
        up_1 = self.up_conv_1(b, trans_4)
        up_2 = self.up_conv_2(up_1, trans_3)
        up_3 = self.up_conv_3(up_2, trans_2)
        up_4 = self.up_conv_4(up_3, trans_1)
        
        # Output layer
        out = self.out(up_4)
        return out
    
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
############ UNET with TR/Spectra ############
##############################################

class UNeTransformedSpectral(nn.Module):
    def __init__(self, in_channels, out_channels, num_bands, spectral_dim=256, num_tokens=196, patch_size=4):
        super(UNeTransformedSpectral, self).__init__()
        
        # spectral embedding
        self.spectral_embed = SpectralEmbedding(input_dim=num_bands, embed_dim=spectral_dim, num_tokens=num_tokens)
        
        # encoder
        self.down_conv_1 = Downsample(in_channels, out_channels=64)
        self.trans_1 = TransformerBlockSpectral(embed_dim=64, signal_dim=spectral_dim, num_heads=2, ff_dim=256, patch_size=patch_size)
        self.down_conv_2 = Downsample(in_channels=64, out_channels=128)
        self.trans_2 = TransformerBlockSpectral(embed_dim=128, signal_dim=spectral_dim, num_heads=2, ff_dim=512, patch_size=patch_size)
        self.down_conv_3 = Downsample(in_channels=128, out_channels=256)
        self.trans_3 = TransformerBlockSpectral(embed_dim=256, signal_dim=spectral_dim, num_heads=4, ff_dim=1024, patch_size=patch_size)
        self.down_conv_4 = Downsample(in_channels=256, out_channels=512)
        self.trans_4 = TransformerBlockSpectral(embed_dim=512, signal_dim=spectral_dim, num_heads=4, ff_dim=2048, patch_size=patch_size)
        
        # bottleneck
        self.bottle_neck = DoubleConv(in_channels=512, out_channels=1024)
        
        # decoder
        self.up_conv_1 = Upsample(in_channels=1024, out_channels=512)
        self.up_conv_2 = Upsample(in_channels=512, out_channels=256)
        self.up_conv_3 = Upsample(in_channels=256, out_channels=128)
        self.up_conv_4 = Upsample(in_channels=128, out_channels=64)
        
        # output layer
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x, spectral_signal, get_weights=False, mask=None):
        
        # create spectral embedding
        spectral_signal = self.spectral_embed(spectral_signal)
        
        # decoder
        down_1, p1 = self.down_conv_1(x)
        trans_1, self_weights_1, cross_weights_1 = self.trans_1(down_1, spectral_signal, mask=mask)
        down_2, p2 = self.down_conv_2(p1)
        trans_2, self_weights_2, cross_weights_2 = self.trans_2(down_2, spectral_signal, mask=mask)
        down_3, p3 = self.down_conv_3(p2)
        trans_3, self_weights_3, cross_weights_3 = self.trans_3(down_3, spectral_signal, mask=mask)
        down_4, p4 = self.down_conv_4(p3)
        trans_4, self_weights_4, cross_weights_4 = self.trans_4(down_4, spectral_signal, mask=mask)
        
        # bottleneck
        b = self.bottle_neck(p4)
        
        # encoder
        up_1 = self.up_conv_1(b, trans_4)
        up_2 = self.up_conv_2(up_1, trans_3)
        up_3 = self.up_conv_3(up_2, trans_2)
        up_4 = self.up_conv_4(up_3, trans_1)
        
        # output layer
        out = self.out(up_4)
        
        if get_weights:
            attn_weights = {
                "self": [self_weights_1, self_weights_2, self_weights_3, self_weights_4],
                "cross": [cross_weights_1, cross_weights_2, cross_weights_3, cross_weights_4]
            }
            return out, attn_weights

        return out