# Adapted from https://github.com/SonyCSLParis/music2latent/blob/master/music2latent/models.py

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def zero_init(module, init_as_zero: bool = True):
    if init_as_zero:
        for p in module.parameters():
            p.detach().zero_()
    return module

def upsample_1d(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest")

def downsample_1d(x):
    return F.avg_pool1d(x, kernel_size=2, stride=2)

def upsample_2d(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest")

def downsample_2d(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)

class FreqGain(nn.Module):
    def __init__(self, freq_dim):
        super(FreqGain, self).__init__()
        self.scale = nn.Parameter(torch.ones((1,1,freq_dim,1)))

    def forward(self, input):
        return input*self.scale

# adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
class GaussianFourierProjection(torch.nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  def __init__(self, embedding_size=128, scale=0.02):
    super().__init__()
    self.W = torch.nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2. * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_size=128, max_positions=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions
    def forward(self, x):
        freqs = torch.arange(start=0, end=self.embedding_size//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x

class MultiheadAttention(nn.MultiheadAttention):
    def _reset_parameters(self):
        super()._reset_parameters()
        self.out_proj = zero_init(self.out_proj)

class ScriptedAttention(nn.Module):
    def __init__(self, dim, heads=4, normalize=True, use_2d=False):
        super(ScriptedAttention, self).__init__()
        
        self.normalize = normalize
        self.use_2d = use_2d
        
        self.mha = MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=0.0, add_zero_attn=False, batch_first=True)
        self.norm = nn.GroupNorm(min(dim//4, 32), dim) if normalize else nn.Identity()

    def forward(self, x):
        inp = x
        x = self.norm(x)
        
        if self.use_2d:
            x = x.permute(0,3,2,1) # shape: [bs,len,freq,channels]
            bs,len,freq,channels = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
            x = x.reshape(bs*len,freq,channels) # shape: [bs*len,freq,channels]
        else:
            bs,len,freq,channels = x.shape[0],x.shape[1],0,x.shape[2]
            x = x.permute(0,2,1) # shape: [bs,len,channels]
        
        x = self.mha(x, x, x, need_weights=False)[0]
        if self.use_2d:
            x = x.reshape(bs,len,freq,channels).permute(0,3,2,1)
        else:
            x = x.permute(0,2,1)
        x = x+inp
        return x
    
class ScriptedConv2d(nn.Conv2d):
    """Custom Conv2d class with proper type annotations for TorchScript"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._conv_forward(x[0], self.weight, self.bias)

class ScriptedUpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=True, normalize=False):
        super(ScriptedUpsampleConv, self).__init__()
        self.normalize = normalize
        self.use_2d = use_2d
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
    def forward(self, x):
        x = self.norm(x)
        if self.use_2d:
            x = upsample_2d(x)
        else:
            x = upsample_1d(x)
        x = self.c(x)
        return x
    
class ScriptedUpsampleConvProj(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=True, normalize=False):
        super(ScriptedUpsampleConvProj, self).__init__()
        self.normalize = normalize
        self.use_2d = use_2d
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.norm(x[0])
        if self.use_2d:
            x = upsample_2d(x)
        else:
            x = upsample_1d(x)
        y = self.c(x)
        return y
    
class ScriptedDownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(ScriptedDownsampleConv, self).__init__()
        self.normalize = normalize
        if out_channels is None:
            out_channels = in_channels
        self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels) if normalize else nn.Identity()
        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='zeros')
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='zeros')

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
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='zeros')
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='zeros')

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.norm(x[0])
        y = self.c(x)
        return y

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
        y = self.c(x)
        return y

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
        y = self.c(x)
        return y

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
        y = self.c(x)
        return y
    
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
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', padding_mode='zeros')
        self.conv2 = zero_init(Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', padding_mode='zeros'))
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
        self.att = ScriptedAttention(out_channels, heads, use_2d=use_2d) if attention else nn.Identity()

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
    def __init__(self, in_channels, out_channels, hparams, cond_channels=None, kernel_size=3, downsample=False, upsample=False, normalize=True, leaky=False, attention=False, heads=4, use_2d=True, normalize_residual=False):
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
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', padding_mode='zeros')
        self.conv2 = zero_init(Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', padding_mode='zeros'))
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
        self.att = ScriptedAttention(out_channels, heads, use_2d=use_2d) if attention else nn.Identity()

        self.min_res_dropout = hparams.min_res_dropout

    def forward(self, x_tup: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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
        self.conv_inp = Conv(channels, input_channels, kernel_size=3, stride=1, padding=1, padding_mode='zeros')

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
        # print("x shape after prenorm_1d_to_2d:", x.shape)
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
