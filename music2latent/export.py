# from .hparams import hparams
from .utils import *
from .scripted_audio import *

from .models import *

from . import nn_diffusion_tilde

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScriptedConv2d(nn.Conv2d):
    """Custom Conv2d class with proper type annotations for TorchScript"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._conv_forward(x[0], self.weight, self.bias)

def upsample_1d(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest")

def downsample_1d(x):
    return F.avg_pool1d(x, kernel_size=2, stride=2)

def upsample_2d(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest")

def downsample_2d(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)

class ScriptedUpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(ScriptedUpsampleConv, self).__init__()
        self.normalize = normalize
        self.use_2d = use_2d
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
    def forward(self, x):
        x = self.norm(x)
        if self.use_2d:
            x = upsample_2d(x)
        else:
            x = upsample_1d(x)
        x = self.c(x)
        return x
    
class ScriptedUpsampleConvProj(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(ScriptedUpsampleConvProj, self).__init__()
        self.normalize = normalize
        self.use_2d = use_2d
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.norm(x[0])
        if self.use_2d:
            x = upsample_2d(x)
        else:
            x = upsample_1d(x)
        x = self.c(x)
        return x
    
class ScriptedDownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(ScriptedDownsampleConv, self).__init__()
        self.normalize = normalize
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.c(x)
        return x
    
class ScriptedDownsampleConvProj(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(ScriptedDownsampleConvProj, self).__init__()
        self.normalize = normalize
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.norm(x[0])
        x = self.c(x)
        return x

class ScriptedUpsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(ScriptedUpsampleFreqConv, self).__init__()
        self.normalize = normalize
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), stride=1, padding='same')

    def forward(self, x):
        x = self.norm(x)
        x = F.interpolate(x, scale_factor=(4.0,1.0), mode="nearest")
        x = self.c(x)
        return x

class ScriptedUpsampleFreqConvProj(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(ScriptedUpsampleFreqConvProj, self).__init__()
        self.normalize = normalize
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), stride=1, padding='same')

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.norm(x[0])
        x = F.interpolate(x, scale_factor=(4.0,1.0), mode="nearest")
        x = self.c(x)
        return x

class ScriptedDownsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(ScriptedDownsampleFreqConv, self).__init__()
        self.normalize = normalize
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), stride=(4,1), padding=(2,0))

    def forward(self, x):
        x = self.norm(x)
        x = self.c(x)
        return x
    
class ScriptedDownsampleFreqConvProj(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(ScriptedDownsampleFreqConvProj, self).__init__()
        self.normalize = normalize
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), stride=(4,1), padding=(2,0))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.norm(x[0])
        x = self.c(x)
        return x
    

class EmbeddingProjector(nn.Module):
    def __init__(self, cond_channels, out_channels):
        super(EmbeddingProjector, self).__init__()
        self.emb_proj = nn.Sequential(nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU())
        self.proj_emb = zero_init(nn.Linear(cond_channels, out_channels))

    def forward(self, x):
        return x
    
class ScriptedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hparams, cond_channels=None, kernel_size=3, downsample=False, upsample=False, normalize=True, leaky=False, attention=False, heads=4, use_2d=False, normalize_residual=False):
        super(ScriptedResBlock, self).__init__()
        self.normalize = normalize
        self.attention = attention
        self.upsample = upsample
        self.downsample = downsample
        self.leaky = leaky
        self.kernel_size = kernel_size
        self.normalize_residual = normalize_residual
        self.use_2d = use_2d
        if use_2d:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv1d
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = zero_init(Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'))
        if in_channels!=out_channels:
            self.res_conv = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = nn.Identity()
        if normalize:
            self.norm1 = nn.GroupNorm(min(in_channels//4, 32), in_channels)
            self.norm2 = nn.GroupNorm(min(out_channels//4, 32), out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        if leaky:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = nn.SiLU()
        if cond_channels is not None:
            self.use_time_emb = True
            self.proj_emb = zero_init(nn.Linear(cond_channels, out_channels))
        else:
            self.use_time_emb = False
            self.proj_emb = nn.Identity()
        self.dropout = nn.Dropout(hparams.dropout_rate)
        if attention:
            self.att = Attention(out_channels, heads, use_2d=use_2d)
        else:
            self.att = nn.Identity()

        self.min_res_dropout = hparams.min_res_dropout

    def forward(self, x_in):
        x = self.norm1(x_in)

        if self.normalize_residual:
            y = x.clone()
        else:
            y = x_in.clone()

        x = self.activation(x)
        if self.downsample:
            if self.use_2d:
                x = downsample_2d(x)
                y = downsample_2d(y)
            else:
                x = downsample_1d(x)
                y = downsample_1d(y)
        if self.upsample:
            if self.use_2d:
                x = upsample_2d(x)
                y = upsample_2d(y)
            else:
                x = upsample_1d(x)
                y = upsample_1d(y)
        x = self.conv1(x)
        # if time_emb is not None:
        #     if self.use_2d:
        #         x = x+self.proj_emb(time_emb)[:,:,None,None]
        #     else:
        #         x = x+self.proj_emb(time_emb)[:,:,None]
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        if x.shape[-1]<=self.min_res_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        y = self.res_conv(y)
        x = x+y
        
        x = self.att(x)

        return x

class ScriptedResBlockProj(nn.Module):
    def __init__(self, in_channels, out_channels, hparams, cond_channels=None, kernel_size=3, downsample=False, upsample=False, normalize=True, leaky=False, attention=False, heads=4, use_2d=False, normalize_residual=False):
        super(ScriptedResBlockProj, self).__init__()
        self.normalize = normalize
        self.attention = attention
        self.upsample = upsample
        self.downsample = downsample
        self.leaky = leaky
        self.kernel_size = kernel_size
        self.normalize_residual = normalize_residual
        self.use_2d = use_2d
        if use_2d:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv1d
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = zero_init(Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'))
        if in_channels!=out_channels:
            self.res_conv = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = nn.Identity()
        if normalize:
            self.norm1 = nn.GroupNorm(min(in_channels//4, 32), in_channels)
            self.norm2 = nn.GroupNorm(min(out_channels//4, 32), out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        if leaky:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = nn.SiLU()
        if cond_channels is not None:
            self.use_time_emb = True
            self.proj_emb = zero_init(nn.Linear(cond_channels, out_channels))
        else:
            self.use_time_emb = False
            self.proj_emb = nn.Identity()
        self.dropout = nn.Dropout(hparams.dropout_rate)
        if attention:
            self.att = Attention(out_channels, heads, use_2d=use_2d)
        else:
            self.att = nn.Identity()

        self.min_res_dropout = hparams.min_res_dropout

    def forward(self, x_tup: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    # def forward(self, x_in, time_emb):
        
        x_in = x_tup[0]
        time_emb = x_tup[1]

        x = self.norm1(x_in)

        if self.normalize_residual:
            y = x.clone()
        else:
            y = x_in.clone()

        x = self.activation(x)
        if self.downsample:
            if self.use_2d:
                x = downsample_2d(x)
                y = downsample_2d(y)
            else:
                x = downsample_1d(x)
                y = downsample_1d(y)
        if self.upsample:
            if self.use_2d:
                x = upsample_2d(x)
                y = upsample_2d(y)
            else:
                x = upsample_1d(x)
                y = upsample_1d(y)
        x = self.conv1(x)
        # if time_emb is not None:
        if self.use_2d:
            x = x+self.proj_emb(time_emb)[:,:,None,None]
        else:
            x = x+self.proj_emb(time_emb)[:,:,None]
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        if x.shape[-1]<=self.min_res_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        y = self.res_conv(y)
        x = x+y
        
        x = self.att(x)

        return x

class ScriptedEncoder(nn.Module):
    def __init__(self, hparams):
        super(ScriptedEncoder, self).__init__()
        
        layers_list = hparams.layers_list_encoder
        attention_list = hparams.attention_list_encoder
        self.layers_list = layers_list
        self.multipliers_list = hparams.multipliers_list
        input_channels = hparams.base_channels*hparams.multipliers_list[0]
        Conv = nn.Conv2d
        self.gain = FreqGain(freq_dim=hparams.hop*2)

        channels = hparams.data_channels
        self.conv_inp = Conv(channels, input_channels, kernel_size=3, stride=1, padding=1)

        self.freq_dim = (hparams.hop*2)//(4**hparams.freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**hparams.freq_downsample_list.count(0))
        
        # DOWNSAMPLING
        down_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(layers_list,hparams.multipliers_list)):
            output_channels = hparams.base_channels*multiplier
            for num in range(num_layers):
                down_layers.append(ScriptedResBlock(input_channels, output_channels, hparams, normalize=hparams.normalization, attention=attention_list[i]==1, heads=hparams.heads, use_2d=True))
                input_channels = output_channels
            if i!=(len(layers_list)-1):
                if hparams.freq_downsample_list[i]==1:
                    down_layers.append(ScriptedDownsampleFreqConv(input_channels, normalize=hparams.pre_normalize_downsampling_encoder))
                else:
                    down_layers.append(ScriptedDownsampleConv(input_channels, use_2d=True, normalize=hparams.pre_normalize_downsampling_encoder))

        if hparams.pre_normalize_2d_to_1d:
            self.prenorm_1d_to_2d = nn.GroupNorm(min(input_channels//4, 32), input_channels)
        else:
            self.prenorm_1d_to_2d = nn.Identity()

        bottleneck_layers = []
        output_channels = hparams.bottleneck_base_channels
        bottleneck_layers.append(nn.Conv1d(input_channels*self.freq_dim, output_channels, kernel_size=1, stride=1, padding='same'))
        for i in range(hparams.num_bottleneck_layers):
            bottleneck_layers.append(ScriptedResBlock(output_channels, output_channels, hparams, normalize=hparams.normalization, use_2d=False))
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)

        self.norm_out = nn.GroupNorm(min(output_channels//4, 32), output_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = nn.Conv1d(output_channels, hparams.bottleneck_channels, kernel_size=1, stride=1, padding='same')
        self.activation_bottleneck = nn.Tanh()
            
        self.down_layers = nn.ModuleList(down_layers)

        # hparams
        self.frequency_scaling = hparams.frequency_scaling

    def forward(self, x):

        x = self.conv_inp(x)
        if self.frequency_scaling:
            x = self.gain(x)
        
        # DOWNSAMPLING
        l = 0
        k = 0
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            l = l + 1
            # print(k)
            if k < len(self.layers_list):
                if l == self.layers_list[k]:
                    k = k + 1
                    l = 0

        x = self.prenorm_1d_to_2d(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))

        for layer in self.bottleneck_layers:
            x = layer(x)
                
        x = self.norm_out(x)
        x = self.activation_out(x)
        x = self.conv_out(x)
        x = self.activation_bottleneck(x)

        return x
    

class ScriptedDecoder(nn.Module):
    def __init__(self, hparams):
        super(ScriptedDecoder, self).__init__()
        
        layers_list = hparams.layers_list_encoder
        attention_list = hparams.attention_list_encoder
        self.layers_list = hparams.layers_list_encoder
        self.reversed_layers_list = list(reversed(layers_list))
        self.multipliers_list = hparams.multipliers_list
        input_channels = hparams.base_channels*hparams.multipliers_list[-1]

        output_channels = hparams.bottleneck_base_channels
        self.conv_inp = nn.Conv1d(hparams.bottleneck_channels, output_channels, kernel_size=1, stride=1, padding='same')
        
        self.freq_dim = (hparams.hop*2)//(4**hparams.freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**hparams.freq_downsample_list.count(0))

        bottleneck_layers = []
        for i in range(hparams.num_bottleneck_layers):
            bottleneck_layers.append(ScriptedResBlock(output_channels, output_channels, hparams, cond_channels = hparams.cond_channels, normalize=hparams.normalization, use_2d=False))

        self.conv_out_bottleneck = nn.Conv1d(output_channels, input_channels*self.freq_dim, kernel_size=1, stride=1, padding='same')
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(hparams.multipliers_list))[1:]+list(reversed(hparams.multipliers_list))[:1]
        freq_upsample_list = list(reversed(hparams.freq_downsample_list))
        up_layers = []      
        for i, (num_layers,multiplier) in enumerate(zip(reversed(layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(ScriptedResBlock(input_channels, input_channels, hparams, normalize=hparams.normalization, attention=list(reversed(attention_list))[i]==1, heads=hparams.heads, use_2d=True))
            if i!=(len(layers_list)-1):
                output_channels = hparams.base_channels*multiplier
                if freq_upsample_list[i]==1:
                    up_layers.append(ScriptedUpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(ScriptedUpsampleConv(input_channels, output_channels, use_2d=True))
                input_channels = output_channels
            
        self.up_layers = nn.ModuleList(up_layers)


    def forward(self, x):

        x = self.conv_inp(x)
        
        for layer in self.bottleneck_layers:
            x = layer(x)
        x = self.conv_out_bottleneck(x)

        x_ls = torch.chunk(x.unsqueeze(-2), self.freq_dim, -3)
        x = torch.cat(x_ls, -2)
        
        # UPSAMPLING
        k = 0
        l = 0
        pyramid_list = []
        for i, layer in enumerate(self.up_layers):
            x = layer(x)
            l = l + 1
            if k < len(self.layers_list):
                if l == 1:
                    pyramid_list.append(x)
                if l == self.layers_list[k] + 1:
                    k = k + 1
                    l = 0

        pyramid_list = pyramid_list[::-1]

        return pyramid_list


class ScriptedUNet(nn_diffusion_tilde.Module):
    __constants__ = ['mods']

    def __init__(self, hparams, sigma_rescale):
        super(ScriptedUNet, self).__init__()
        
        self.layers_list = hparams.layers_list
        self.reversed_layers_list = list(reversed(hparams.layers_list))
        self.multipliers_list = hparams.multipliers_list
        input_channels = hparams.base_channels*hparams.multipliers_list[0]
        Conv = nn.Conv2d

        self.encoder = ScriptedEncoder(hparams)
        self.decoder = ScriptedDecoder(hparams)

        if hparams.use_fourier:
            self.emb = GaussianFourierProjection(embedding_size=hparams.cond_channels, scale=hparams.fourier_scale)
        else:
            self.emb = PositionalEmbedding(embedding_size=hparams.cond_channels)

        self.emb_proj = nn.Sequential(nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU())

        self.scale_inp = nn.Sequential(nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), zero_init(nn.Linear(hparams.cond_channels, hparams.hop*2)))
        self.scale_out = nn.Sequential(nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), zero_init(nn.Linear(hparams.cond_channels, hparams.hop*2)))

        self.conv_inp = Conv(hparams.data_channels, input_channels, kernel_size=3, stride=1, padding=1)
        
        # DOWNSAMPLING
        down_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(hparams.layers_list,hparams.multipliers_list)):
            output_channels = hparams.base_channels*multiplier
            for num in range(num_layers):
                down_layers.append(ScriptedConv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0))
                down_layers.append(ScriptedResBlockProj(output_channels, output_channels, hparams, cond_channels = hparams.cond_channels, normalize=hparams.normalization, attention=hparams.attention_list[i]==1, heads=hparams.heads, use_2d=True))                
                input_channels = output_channels
            if i!=(len(hparams.layers_list)-1):
                output_channels = hparams.base_channels*hparams.multipliers_list[i+1]
                if hparams.freq_downsample_list[i]==1:
                    down_layers.append(ScriptedDownsampleFreqConvProj(input_channels, output_channels))
                else:
                    down_layers.append(ScriptedDownsampleConvProj(input_channels, output_channels, use_2d=True))

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(hparams.multipliers_list))[1:]+list(reversed(hparams.multipliers_list))[:1]
        freq_upsample_list = list(reversed(hparams.freq_downsample_list))
        up_layers = []      
        for i, (num_layers,multiplier) in enumerate(zip(reversed(hparams.layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(ScriptedConv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0))
                up_layers.append(ScriptedResBlockProj(input_channels, input_channels, hparams, cond_channels = hparams.cond_channels, normalize=hparams.normalization, attention=list(reversed(hparams.attention_list))[i]==1, heads=hparams.heads, use_2d=True))
            if i!=(len(hparams.layers_list)-1):
                output_channels = hparams.base_channels*multiplier
                if freq_upsample_list[i]==1:
                    up_layers.append(ScriptedUpsampleFreqConvProj(input_channels, output_channels))
                else:
                    up_layers.append(ScriptedUpsampleConvProj(input_channels, output_channels, use_2d=True))
                input_channels = output_channels
                
        self.conv_decoded = Conv(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.norm_out = nn.GroupNorm(min(input_channels//4, 32), input_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = zero_init(Conv(input_channels, hparams.data_channels, kernel_size=3, stride=1, padding=1))
            
        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

        # hparams
        self.sigma_max = hparams.sigma_max
        self.frequency_scaling = hparams.frequency_scaling

        self.data_channels = hparams.data_channels
        freq_downsample_list = hparams.freq_downsample_list
        self.downscaling_factor = 2**freq_downsample_list.count(0)

        self.sigma_correct = hparams.sigma_min
        self.sigma_data = hparams.sigma_data

        self.hop = hparams.hop

        self.sqrt2 = np.sqrt(2.)
        self.sqrt3 = np.sqrt(3.)

        self.sigma_rescale = sigma_rescale

        self.eval()
        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=1,
            out_channels=1,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, 1 + 1)],
            output_labels=['(signal) Channel %d'%d for d in range(1, 1+1)],
            test_buffer_size = 42496
        )

    def get_c(self, sigma, sigma_correct: float, sigma_data: float):
        c_skip = (sigma_data**2.)/(((sigma-sigma_correct)**2.) + (sigma_data**2.))
        c_out = (sigma_data*(sigma-sigma_correct))/(((sigma_data**2.) + (sigma**2.))**0.5)
        c_in = 1./(((sigma**2.)+(sigma_data**2.))**0.5)
        return c_skip.reshape(-1,1,1,1), c_out.reshape(-1,1,1,1), c_in.reshape(-1,1,1,1)
    
    def forward_generator(self, latents, x):

        sigma = self.sigma_max
        
        inp = x
        
        # CONDITIONING
        sigma = torch.ones((x.shape[0],), dtype=torch.float32).to(x.device)*sigma
        sigma_log = torch.log(sigma)/4.
        emb_sigma_log = self.emb(sigma_log)
        time_emb = self.emb_proj(emb_sigma_log)

        scale_w_inp = self.scale_inp(emb_sigma_log).reshape(x.shape[0],1,-1,1)
        scale_w_out = self.scale_out(emb_sigma_log).reshape(x.shape[0],1,-1,1)
            
        c_skip, c_out, c_in = self.get_c(sigma, self.sigma_correct, self.sigma_data)
        
        x = c_in*x

        if latents.shape == x.shape:
            latents = self.encoder(latents)

        # if pyramid_latents is None:
        pyramid_latents = self.decoder(latents)

        x = self.conv_inp(x)
        if self.frequency_scaling:
            x = (1.+scale_w_inp)*x
        
        skip_list = []
        
        # DOWNSAMPLING
        k = 0
        l = 0
        for i, layer in enumerate(self.down_layers):
            l = l + 1
            if l % self.layers_list[k] == 1 and l != self.layers_list[k]*2+1:
                d = layer((pyramid_latents[k],time_emb))
                x = (x+d)/self.sqrt2
            elif l % self.layers_list[k] == 0 and l != self.layers_list[k]*2+1:
                x = layer((x, time_emb))
                skip_list.append(x)
            elif l == self.layers_list[k]*2+1 and i != (len(self.down_layers)-1):
                x = layer((x,time_emb))
                k = k + 1
                l = 0
        # x, skip_list = self.down_layers(x, time_emb, pyramid_latents, self.layers_list)

        # UPSAMPLING
        k = 0
        l = 0
        for i, layer in enumerate(self.up_layers):
            l = l + 1
            if l % self.layers_list[k] == 1 and l != self.layers_list[k]*2+1:
                d = layer((pyramid_latents[-k-1],time_emb))
                x = (x+skip_list.pop()+d)/self.sqrt3
            elif l % self.layers_list[k] == 0 and l != self.layers_list[k]*2+1:
                x = layer((x, time_emb)) 
            elif l == self.layers_list[k]*2+1 and i != (len(self.up_layers)-1):
                x = layer((x,time_emb))
                k = k + 1
                l = 0
                
        d = self.conv_decoded(pyramid_latents[0])
        x = (x+d)/self.sqrt2

        x = self.norm_out(x)
        x = self.activation_out(x)
        if self.frequency_scaling:
            x = (1.+scale_w_out)*x
        x = self.conv_out(x)

        out = c_skip*inp + c_out*x 

        return out
    
    @torch.jit.export
    def decode(self, latent):
        latent = latent*self.sigma_rescale

        num_samples = latent.shape[0]
        
        sample_length = int(latent.shape[-1]*self.downscaling_factor)

        init_noise = torch.randn((num_samples, self.data_channels, self.hop*2, sample_length)).to(latent.device)*self.sigma_max
        decoded_spec = self.forward_generator(latent, init_noise)
        decoded_wv = realimag2wv(decoded_spec, self.hop, fac=4)
        return decoded_wv
    
    # def forward(self, data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas):
    @torch.jit.export
    def forward(self, wv):
        assert wv.dim() == 3, "Input should be a 4D tensor (batch_size, channels, sample)"
        with torch.no_grad():
            repr_encoder = wv2realimag(wv[0], self.hop, fac=4)
            latent = self.encoder(repr_encoder)
            latent = latent/self.sigma_rescale

            wv_rec = self.decode(latent).unsqueeze(0)  # Add batch dimension
        assert wv_rec.dim() == 3, "Output should be a 4D tensor (batch_size, channels, sample)"
        return wv_rec