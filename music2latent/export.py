from .hparams import hparams
from .utils import *
from .audio import *

from .models import *

from . import nn_diffusion_tilde


class ScriptedUNet(nn_diffusion_tilde.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.layers_list = hparams.layers_list
        self.multipliers_list = hparams.multipliers_list
        input_channels = hparams.base_channels*hparams.multipliers_list[0]
        Conv = nn.Conv2d

        self.encoder = Encoder()
        self.decoder = Decoder()

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
                down_layers.append(Conv(output_channels, output_channels, kernel_size=1, stride=1, padding=0))
                down_layers.append(ResBlock(output_channels, output_channels, hparams.cond_channels, normalize=hparams.normalization, attention=hparams.attention_list[i]==1, heads=hparams.heads, use_2d=True))                
                input_channels = output_channels
            if i!=(len(hparams.layers_list)-1):
                output_channels = hparams.base_channels*hparams.multipliers_list[i+1]
                if hparams.freq_downsample_list[i]==1:
                    down_layers.append(DownsampleFreqConv(input_channels, output_channels))
                else:
                    down_layers.append(DownsampleConv(input_channels, output_channels, use_2d=True))

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(hparams.multipliers_list))[1:]+list(reversed(hparams.multipliers_list))[:1]
        freq_upsample_list = list(reversed(hparams.freq_downsample_list))
        up_layers = []      
        for i, (num_layers,multiplier) in enumerate(zip(reversed(hparams.layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(Conv(input_channels, input_channels, kernel_size=1, stride=1, padding=0))
                up_layers.append(ResBlock(input_channels, input_channels, hparams.cond_channels, normalize=hparams.normalization, attention=list(reversed(hparams.attention_list))[i]==1, heads=hparams.heads, use_2d=True))
            if i!=(len(hparams.layers_list)-1):
                output_channels = hparams.base_channels*multiplier
                if freq_upsample_list[i]==1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(UpsampleConv(input_channels, output_channels, use_2d=True))
                input_channels = output_channels
                
        self.conv_decoded = Conv(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.norm_out = nn.GroupNorm(min(input_channels//4, 32), input_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = zero_init(Conv(input_channels, hparams.data_channels, kernel_size=3, stride=1, padding=1))
            
        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)


    def forward_generator(self, latents, x, sigma=None, pyramid_latents=None):

        if sigma is None:
            sigma = hparams.sigma_max
        
        inp = x
        
        # CONDITIONING
        sigma = torch.ones((x.shape[0],), dtype=torch.float32).to(x.device)*sigma
        sigma_log = torch.log(sigma)/4.
        emb_sigma_log = self.emb(sigma_log)
        time_emb = self.emb_proj(emb_sigma_log)

        scale_w_inp = self.scale_inp(emb_sigma_log).reshape(x.shape[0],1,-1,1)
        scale_w_out = self.scale_out(emb_sigma_log).reshape(x.shape[0],1,-1,1)
            
        c_skip, c_out, c_in = get_c(sigma)
        
        x = c_in*x

        if latents.shape == x.shape:
            latents = self.encoder(latents)

        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents)

        x = self.conv_inp(x)
        if hparams.frequency_scaling:
            x = (1.+scale_w_inp)*x
        
        skip_list = []
        
        # DOWNSAMPLING
        k = 0
        r = 0
        for i,num_layers in enumerate(self.layers_list):
            for num in range(num_layers):
                d = self.down_layers[k](pyramid_latents[i])
                k = k+1
                x = (x+d)/np.sqrt(2.)
                x = self.down_layers[k](x, time_emb)
                skip_list.append(x)
                k = k+1
            if i!=(len(self.layers_list)-1):
                x = self.down_layers[k](x)
                k = k+1
              
        # UPSAMPLING
        k = 0
        for i,num_layers in enumerate(reversed(self.layers_list)):
            for num in range(num_layers):
                d = self.up_layers[k](pyramid_latents[-i-1])
                k = k+1
                x = (x+skip_list.pop()+d)/np.sqrt(3.)
                x = self.up_layers[k](x, time_emb)
                k = k+1
            if i!=(len(self.layers_list)-1):
                x = self.up_layers[k](x)
                k = k+1
                
        d = self.conv_decoded(pyramid_latents[0])
        x = (x+d)/np.sqrt(2.)

        x = self.norm_out(x)
        x = self.activation_out(x)
        if hparams.frequency_scaling:
            x = (1.+scale_w_out)*x
        x = self.conv_out(x)

        out = c_skip*inp + c_out*x 

        return out
    

    def forward(self, data_encoder, noisy_samples, noisy_samples_plus_one, sigmas_step, sigmas):
        latents = self.encoder(data_encoder)
        pyramid_latents = self.decoder(latents)
        fdata = self.forward_generator(latents, noisy_samples, sigmas_step, pyramid_latents).detach()
        fdata_plus_one = self.forward_generator(latents, noisy_samples_plus_one, sigmas, pyramid_latents)
        return fdata, fdata_plus_one