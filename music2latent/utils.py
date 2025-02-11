import os
from os import path as ospath
from itertools import islice
from tqdm import tqdm
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import torch

from .hparams import hparams
from .audio import *

# Get scaling coefficients c_skip, c_out, c_in based on noise sigma
# These are used to scale the input and output of the consistency model, while satisfying the boundary condition for consistency models
# Parameters:
# sigma: noise level
# Returns:
# c_skip, c_out, c_in: scaling coefficients
def get_c(sigma):
    sigma_correct = hparams.sigma_min
    c_skip = (hparams.sigma_data**2.)/(((sigma-sigma_correct)**2.) + (hparams.sigma_data**2.))
    c_out = (hparams.sigma_data*(sigma-sigma_correct))/(((hparams.sigma_data**2.) + (sigma**2.))**0.5)
    c_in = 1./(((sigma**2.)+(hparams.sigma_data**2.))**0.5)
    return c_skip.reshape(-1,1,1,1), c_out.reshape(-1,1,1,1), c_in.reshape(-1,1,1,1)

# Get noise level sigma_i based on index i and number of discretization steps k
# Parameters:
# i: index
# k: number of discretization steps
# Returns:
# sigma_i: noise level corresponding to index i
def get_sigma(i, k):
    return (hparams.sigma_min**(1./hparams.rho) + ((i-1)/(k-1))*(hparams.sigma_max**(1./hparams.rho)-hparams.sigma_min**(1./hparams.rho)))**hparams.rho

# Get noise level sigma for a continuous index i in [0, 1]
# Follows parameterization in https://openreview.net/pdf?id=FmqFfMTNnv
# Parameters:
# i: continuous index in [0, 1]
# Returns:
# sigma: corresponding noise level
def get_sigma_continuous(i):
    return (hparams.sigma_min**(1./hparams.rho) + i*(hparams.sigma_max**(1./hparams.rho)-hparams.sigma_min**(1./hparams.rho)))**hparams.rho


# Get noise level sigma_{i-step} where i is a continuous index in (0, 1]
# Parameters:
# i: index
# step: step to be taken towards lower sigma
# Returns:
# sigma_{i-step}: noise level corresponding to i-step
def get_sigma_step_continuous(sigma_i, step):
    return ((sigma_i**(1./hparams.rho) - step*(hparams.sigma_max**(1./hparams.rho)-hparams.sigma_min**(1./hparams.rho)))**hparams.rho).clamp(min=hparams.sigma_min)

# Add Gaussian noise to input x based on given noise and sigma
# Parameters:
# x: input tensor
# noise: tensor containing Gaussian noise
# sigma: noise level
# Returns:
# x_noisy: x with noise added
def add_noise(x, noise, sigma):
    return x + sigma.reshape(-1,1,1,1)*noise

# Get loss weighting factor based on two adjacent sigma values
# Parameters:
# sigma_plus_one: larger sigma value
# sigma: smaller sigma value
# Returns:
# weight: loss weighting factor
def get_loss_weight(sigma_plus_one, sigma):
    return 1./(sigma_plus_one-sigma)

# Get step size for continuous training (instead of discretization steps for discrete training)
# implements an exponential schedule which works better than the cosine one (unpublished)
# Parameters:
# k: training iteration
# Returns:
# step: step size
def get_step_schedule(k):
    if hparams.schedule=='exponential':
        min_step = hparams.base_step**float(hparams.end_exp)
        exp = ((float(k)/float(hparams.total_iters))*(float(hparams.end_exp)-hparams.start_exp))+hparams.start_exp # in the range [1,exp_step]
        step = hparams.base_step**exp
        step = max(step, min_step)
    elif hparams.schedule=='constant':
        step = hparams.base_step**float(hparams.end_exp)
    else:
        raise NameError('schedule must be one of: (exponential, constant)')
    return step

# Get sampling weights for sigma values
# Samples from lognormal distribution
# Parameters:
# k: number of discretization steps
# Returns:
# weights: sampling weights
def get_sampling_weights(k, device='cuda'):
    sigma = get_sigma(torch.linspace(1, k-1, k-1, dtype=torch.int32, device=device), k)
    return gaussian_pdf(torch.log(sigma))

# Get stepped index for continuous sampling
# Parameters:
# inds: continuous indices in [0, 1]
# step: step size
# Returns:
# inds_stepped: indices with given step
def get_step_continuous(inds, step):
    steps = torch.ones_like(inds)*step
    return (inds-steps).clamp(min=0.)

# Reverse the probability flow ODE by one step
# Parameters:
#   x: input
#   noise: Gaussian noise 
#   sigma: noise level
# Returns:
#   x: x after reversing ODE by one step
def reverse_step(x, noise, sigma):
    return x + ((sigma**2 - hparams.sigma_min**2)**0.5)*noise

# Gaussian probability density function, used to sample noise levels with lognormal distribution
# Parameters:
#   x: input 
# Returns:  
#   pdf: probability density at x
def gaussian_pdf(x):
    return (1./(hparams.p_std*(2.*np.pi)**0.5)) * torch.exp(-0.5*((x-hparams.p_mean)/hparams.p_std)**2.)

# Denoise samples at a given noise level
# Parameters:
#   model: consistency model
#   noisy_samples: input noisy samples
#   sigma: noise level
# Returns: 
#   pred_noises: predicted noise
#   pred_samples: denoised samples
def denoise(model, noisy_samples, sigma, latents=None):
    # Denoise samples
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hparams.mixed_precision):
            if latents is not None:
                pred_samples = model.forward_generator(latents, noisy_samples, sigma)
            else:
                pred_samples = model.forward_generator(noisy_samples, sigma)
    # Sample noise
    pred_noises = torch.randn_like(pred_samples)
    return pred_noises, pred_samples

# Reverse the diffusion process to generate samples
# Parameters:
#   model: trained consistency model
#   initial_noise: initial noise to start from 
#   diffusion_steps: number of steps to reverse
# Returns:
#   final_samples: generated samples
def reverse_diffusion(model, initial_noise, diffusion_steps, latents=None):
    next_noisy_samples = initial_noise
    # Reverse process step-by-step
    for k in range(diffusion_steps):

        # Get sigma values
        sigma = get_sigma(diffusion_steps+1-k, diffusion_steps+1)
        next_sigma = get_sigma(diffusion_steps-k, diffusion_steps+1)

        # Denoise 
        noisy_samples = next_noisy_samples
        pred_noises, pred_samples = denoise(model, noisy_samples, sigma, latents)

        # Step to next (lower) noise level
        next_noisy_samples = reverse_step(pred_samples, pred_noises, next_sigma)

    return pred_samples.detach().cpu()

# Generate samples with consistency model
# Parameters:
#   model: trained consistency model 
#   num_samples: number of samples to generate (batch size)
#   diffusion_steps: number of steps
# Returns:
#   generated_images: final generated samples
def generate(model, num_samples=9, diffusion_steps=3, seconds=None, latents=None):
    if seconds is None:
        sample_length = hparams.data_length
    else:
        downscaling_factor = 2**hparams.freq_downsample_list.count(0)
        sample_length = int((((seconds*hparams.sample_rate)//hparams.hop)//downscaling_factor)*downscaling_factor)
    if latents is not None:
        num_samples = latents.shape[0]
        downscaling_factor = 2**hparams.freq_downsample_list.count(0)
        sample_length = int(latents.shape[-1]*downscaling_factor)
    initial_noise = torch.randn((num_samples, hparams.data_channels, hparams.hop*2, sample_length)).cuda()*hparams.sigma_max
    generated_images = reverse_diffusion(model, initial_noise, diffusion_steps, latents=latents)
    return to_waveform(generated_images)

# Encode and Decode samples with consistency model
# Parameters:
#   model: trained consistency model 
#   dataset: dataset of test samples
#   num_samples: number of samples to generate (batch size)
# Returns:
#   generated_samples: final generated samples
@torch.no_grad()
def encode_decode(model, dataset, num_samples=9, diffusion_steps=1):
    device = next(model.parameters()).device
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True, shuffle=True, num_workers=0)
    real = []
    fake = []
    for x in tqdm(islice(dataloader, num_samples)):
        repr_encoder = to_representation_encoder(x.to(device))
        latent = model.encoder(repr_encoder)
        generated_samples = generate(model, diffusion_steps=diffusion_steps, latents=latent)
        real.append(x.squeeze(0).cpu())
        fake.append(generated_samples.squeeze(0).cpu())
    return real, fake

# Generate a batch of samples by splitting into multiple mini-batches that fit in GPU memory
# Parameters:
#   model: trained consistency model
#   num_samples: total samples to generate
#   max_batch_size: max mini-batch size
# Returns:
#   samples: generated samples
def generate_batch(model, num_samples, max_batch_size=1024, diffusion_steps=1):
    print(f'Generating {num_samples} samples...')
    generated_batches = []
    batches = num_samples//max_batch_size
    for b in range(batches):
        generated_batches.append(generate(model, max_batch_size, diffusion_steps))
    remainder = num_samples%max_batch_size
    if remainder!=0:
        generated_batches.append(generate(model, remainder, diffusion_steps))
    return torch.cat(generated_batches, 0)

# Generate a batch of samples by splitting into multiple mini-batches that fit in GPU memory
# Parameters:
#   model: trained consistency model
#   dataset: dataset of test samples
#   num_samples: total samples to generate
#   max_batch_size: max mini-batch size
# Returns:
#   samples: generated samples
@torch.no_grad()
def encode_decode_batch(model, dataset, num_samples, diffusion_steps=1):
    device = next(model.parameters()).device
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True, shuffle=False, num_workers=0)
    print(f'Generating {num_samples} samples...')
    generated_batches = []
    batches = num_samples
    print(f'Generating {num_samples} samples in {batches} batches...')
    for x in tqdm(islice(dataloader, batches+1)):
        repr_encoder = to_representation_encoder(x.to(device))
        latents = model.encoder(repr_encoder)
        generated_samples = generate(model, diffusion_steps=diffusion_steps, latents=latents)
        generated_batches.append(generated_samples.squeeze(0).cpu())
    return generated_batches

# Encode audio sample
# Parameters:
#   audio_path: path of audio sample
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   latent: compressed latent representation with shape [audio_channels, dim, latent_length]
@torch.no_grad()
def encode_audio(audio_path, trainer, device='cuda', return_input=False):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    downscaling_factor = 2**hparams.freq_downsample_list.count(0)
    audio_original, sr = sf.read(audio_path, dtype='float32', always_2d=True)
    audio = np.transpose(audio_original, [1,0]) # [audio_channels, audio_length]
    audio = torch.from_numpy(audio).to(device)
    repr_encoder = to_representation_encoder(audio)
    sample_length = repr_encoder.shape[-1]
    # crop sample to be compatiblen with downscaling factor
    repr_encoder = repr_encoder[:,:,:,:(sample_length//downscaling_factor)*downscaling_factor]
    with trainer.ema.average_parameters():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hparams.mixed_precision):
            latent = trainer.gen.encoder(repr_encoder)
    if return_input:
        return audio_original, sr, latent.cpu().numpy()
    return latent.cpu().numpy()

# Decode latent
# Parameters:
#   latent: numpy array of latent from encoder with shape [audio_channels, dim, latent_length]
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   audio: decoded waveform with shape [audio_length, audio_channels]
@torch.no_grad()
def decode_latent(latent, trainer, diffusion_steps=2, device='cuda'):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    latent = torch.from_numpy(latent).to(device)
    with trainer.ema.average_parameters():
        audio = generate(trainer.gen, diffusion_steps=diffusion_steps, latents=latent)
    return audio.cpu().transpose(1,0).numpy()

# Plot a batch of images in a grid
# Parameters:
#   batch: batch of images   
# Returns: 
#   fig: figure object
#   axs: axes array
def grid5x5(batch):
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(5,5))
    c=0
    for k in range(5):
        for l in range(5):
            # Plot image in position (k,l)
            axs[k][l].imshow(batch[c])
            axs[k][l].axis('off')
            c+=1
    return fig,axs


def is_path(variable):
    return isinstance(variable, str) and os.path.exists(variable)

# Replace NaNs in tensor with zeros 
# Parameters:
#   x: input tensor
# Returns: 
#   x: tensor with NaNs replaced by 0
def clean_nan(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

# Mean squared error between tensors
# Can apply weights and return per-sample or batch average MSE
# Parameters:
#   x, y: input tensors   
#   w: weights tensor (optional)
#   mean: whether to return batch average or per-sample MSE
# Returns:
#   mse: mean squared error
def mse(x,y, w=None, mean=True):
    diff = torch.flatten((x-y)**2, start_dim=1).mean(-1)
    diff = torch.nan_to_num(diff)
    if w is not None:
        diff = diff*w.squeeze()
    if mean:
        return diff.mean()
    else:
        return diff

# Mean absolute error between tensors 
# Can apply weights and return per-sample or batch average MAE
# Parameters:
#   x, y: input tensors
#   w: weights tensor (optional) 
#   mean: whether to return batch average or per-sample MAE
# Returns:
#   mae: mean absolute error
def mae(x,y, w=None, mean=True):
    diff = torch.flatten(torch.abs(x-y), start_dim=1).mean(-1)
    diff = torch.nan_to_num(diff)
    if w is not None:
        diff = diff*w.squeeze()
    if mean:
        return diff.mean()
    else:
        return diff
    
# Pseudo-Huber loss between tensors
# Uses parameters from https://openreview.net/pdf?id=FmqFfMTNnv
# Can apply weights and return per-sample or batch average Pseudo-Huber
# Parameters:  
#   x, y: input tensors
#   w: weights tensor (optional)
#   mean: whether to return batch average or per-sample Pseudo-Huber
#   c: parameter for the 'smoothness' of the loss function
# Returns:
#   loss: Pseudo-Huber loss
def huber(x,y, w=None, mean=True, c=0.00054):
    diff = torch.flatten((x-y)**2, start_dim=1)
    data_dim = diff.shape[-1]
    c = c*torch.sqrt(torch.ones((1,), device=x.device)*data_dim)
    diff = torch.sum(diff, -1)
    diff = torch.sqrt(diff+c**2)-c
    diff = torch.nan_to_num(diff)
    if w is not None:
        diff = diff*w.squeeze()
    if mean:
        return diff.mean()
    else:
        return diff


def is_path(variable):
    return isinstance(variable, str) and os.path.exists(variable)


def get_grad_norm(parameters, norm_type = 2.0):
    """
    Calculate the gradient norm of an iterable of parameters.
    
    Args:
    parameters (Iterable[Tensor]): an iterable of Tensors that will have gradients normalized
    norm_type (float): type of the used p-norm. Can be 'inf' for infinity norm.
    
    Returns:
    Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if len(grads) == 0:
        return torch.tensor(0.)
    
    first_device = grads[0].device
    
    norms = []
    for grad in grads:
            norms.append(torch.linalg.vector_norm(grad, norm_type))
    
    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    
    return total_norm


def download_model():
    filepath = os.path.abspath(__file__)
    lib_root = os.path.dirname(filepath)

    if not ospath.exists(lib_root + "/models/music2latent.pt"):
        print("Downloading model...")
        os.makedirs(lib_root + "/models", exist_ok=True)
        _ = hf_hub_download(repo_id="SonyCSLParis/music2latent", filename="music2latent.pt", cache_dir=lib_root + "/models", local_dir=lib_root + "/models")
        print("Model was downloaded successfully!")