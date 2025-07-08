# from .hparams import hparams

from .scripted_audio import *
from .scripted_models import *

from . import nn_diffusion_tilde

import torch
import torch.nn as nn

class ScriptedUNet(nn_diffusion_tilde.Module):
    __constants__ = ['mods']

    def __init__(self, hparams, sigma_rescale):
        super(ScriptedUNet, self).__init__()
        
        self.layers_list = hparams.layers_list
        self.reversed_layers_list = list(reversed(hparams.layers_list))
        self.multipliers_list = hparams.multipliers_list
        input_channels = hparams.base_channels*hparams.multipliers_list[0]
        # Conv = nn.Conv2d

        self.encoder = ScriptedEncoder(hparams)
        self.decoder = ScriptedDecoder(hparams)

        if hparams.use_fourier:
            self.emb = GaussianFourierProjection(embedding_size=hparams.cond_channels, scale=hparams.fourier_scale)
        else:
            self.emb = PositionalEmbedding(embedding_size=hparams.cond_channels)

        self.emb_proj = nn.Sequential(nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU())
        self.scale_inp = nn.Sequential(nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), zero_init(nn.Linear(hparams.cond_channels, hparams.hop*2)))
        self.scale_out = nn.Sequential(nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), nn.Linear(hparams.cond_channels, hparams.cond_channels), nn.SiLU(), zero_init(nn.Linear(hparams.cond_channels, hparams.hop*2)))

        self.conv_inp = nn.Conv2d(hparams.data_channels, input_channels, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        
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
                
        self.conv_decoded = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.norm_out = nn.GroupNorm(min(input_channels//4, 32), input_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = zero_init(nn.Conv2d(input_channels, hparams.data_channels, kernel_size=3, stride=1, padding=1, padding_mode='zeros'))
            
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

        self.stft_processor = StreamingSTFT(self.hop, 4)
        self.istft_processor = StreamingISTFT(self.hop, 4)

        self.batch_size = 1
        self.max_length = self.downscaling_factor*15

        self.register_buffer('init_noise', torch.randn((self.batch_size, 
                                                        self.data_channels, 
                                                        self.hop*2, 
                                                        self.downscaling_factor*self.max_length)
                                                        )*self.sigma_max)
        
        self.eval()

        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=1,
            out_channels=1,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, 1 + 1)],
            output_labels=['(signal) Channel %d'%d for d in range(1, 1+1)],
            test_buffer_size = 16384
        )
        self.register_method(
            "encode",
            in_channels=1,
            in_ratio=1,
            out_channels=hparams.bottleneck_channels,
            out_ratio=4096,
            input_labels=['(signal) Channel %d'%d for d in range(1, 1 + 1)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(hparams.bottleneck_channels)
            ],
            test_buffer_size = 16384
        )
        self.register_method(
            "decode",
            in_channels=hparams.bottleneck_channels,
            in_ratio=4096,
            out_channels=1,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i+1}'
                for i in range(hparams.bottleneck_channels)
            ],
            output_labels=['(signal) Channel %d'%d for d in range(1, 1+1)],
            test_buffer_size = 16384
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
        assert latent.dim() == 3, "Input should be a 3D tensor (batch_size, latent_dim, sample)"
        with torch.no_grad():
            latent = latent*self.sigma_rescale
            sample_length = int(latent.shape[-1]*self.downscaling_factor)
            
            if self.init_noise.numel() > 0 and self.init_noise.shape[0] != latent.shape[0]:
                self.init_noise = self.init_noise.expand(latent.shape[0], -1).contiguous()

            decoded_spec = self.forward_generator(latent, self.init_noise[..., :sample_length])
            wv_rec = self.istft_processor.process_chunk(decoded_spec).unsqueeze(0)
        assert wv_rec.dim() == 3, "Output should be a 3D tensor (batch_size, channels, sample)"
        return wv_rec
    
    @torch.jit.export
    def encode(self, wv):
        assert wv.dim() == 3, "Input should be a 3D tensor (batch_size, channels, sample)"
        with torch.no_grad():
            repr_encoder = self.stft_processor.process_chunk(wv[0])
            latent = self.encoder(repr_encoder)
            latent = latent/self.sigma_rescale
        assert latent.dim() == 3, "Output should be a 3D tensor (batch_size, latent_dim, sample)"
        return latent
    
    @torch.jit.export
    def forward(self, wv):
        assert wv.dim() == 3, "Input should be a 3D tensor (batch_size, channels, sample)"
        with torch.no_grad():
            repr_encoder = self.stft_processor.process_chunk(wv[0])
            latent = self.encoder(repr_encoder)

            sample_length = int(latent.shape[-1]*self.downscaling_factor)

            if self.init_noise.numel() > 0 and self.init_noise.shape[0] != latent.shape[0]:
                self.init_noise = self.init_noise.expand(latent.shape[0], -1).contiguous()

            decoded_spec = self.forward_generator(latent, self.init_noise[..., :sample_length])
            wv_rec = self.istft_processor.process_chunk(decoded_spec).unsqueeze(0)
        assert wv_rec.dim() == 3, "Output should be a 3D tensor (batch_size, channels, sample)"
        return wv_rec