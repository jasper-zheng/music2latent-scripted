# hparams.py
import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class HParams:
    # TRAINING
    batch_size: int = 16                                                            # batch size
    lr: float = 0.0001                                                              # learning rate
    lr_decay: str = 'cosine'                                                        # learning rate schedule ['cosine', 'linear', 'inverse_sqrt]       
    start_decay_iteration: int = 0                                                  # start decaying learning rate from this iteration
    final_lr: float = 0.000001                                                      # if exponential_lr_decay=True, this is the learning rate after total_iters
    warmup_steps: int = 10000                                                       # number of warmup steps
    accumulate_gradients: int = 1                                                   # will accumulate the gradients from this number of batches befire updating
    total_iters: int = 800000                                                       # total iterations
    iters_per_epoch: int = 10000                                                    # number of iterations approximately in every epoch
    checkpoint_path: str = 'checkpoints'                                            # path where to save config and checkpoints
    torch_compile_cache_dir: str = 'tmp/torch_compile'                              # path where to save compiled kernels
    compile_model: bool = True                                                      # compile the model for faster training (will require ~10 minutes of compilation time only on first run)
    mixed_precision: bool = True                                                    # use mixed precision (float16)
    num_workers: int = 16                                                           # number of dataloader workers
    multi_gpu: bool = False                                                         # use DistributedDataParallel multi-gpu training
    seed: int = 42                                                                  # seed for Pytorch and Numpy
    load_path: Optional[str] = None                                                 # load checkpoint from this path
    load_iter: bool = True                                                          # if False, reset the scheduler and start from iteration 0
    load_ema: bool = True                                                           # if False, do not load the EMA weights from checkpoint
    load_optimizer: bool = True                                                     # if False, do not load the optimizer parameters from checkpoint (helps in case of resuming collapsed run)
    optimizer_beta1: float = 0.9                                                    # beta1 parameter for Adam optimizer
    optimizer_beta2: float = 0.999                                                  # beta2 parameter for Adam optimizer

    # EXPONENTIAL MOVING AVERAGE
    enable_ema: bool = True                                                         # track exponential moving averages for better inference model
    ema_momentum: float = 0.9999                                                    # exponential moving average momentum parameter
    warmup_ema: bool = True                                                         # use warmup for exponential moving average

    # DATA
    data_path_test: str = field(init=False)                                         # path of samples used for FAD testing (e.g. musiccaps)
    data_paths: List[str] = field(init=False)                                       # list of paths of datasets
    data_fractions: Optional[List[float]] = None                                    # list of sampling weights of each dataset (if None, equal sampling weights)
    data_extensions: List[str] = field(default_factory=lambda: ['.wav', '.flac'])   # list of extensions of audio files to search for in the given paths
    rms_min: float = 0.001                                                          # minimum RMS value for audio samples used for training
    data_channels: int = 2                                                          # channels of input data (real-imaginary STFT requires 2)
    data_length: int = 64                                                           # sequence length of input spectrogram
    data_length_test: int = 256                                                     # sequence length of spectrograms used for testing
    sample_rate: int = 48000                                                        # sampling rate used to render audio samples (does not matter for training)
    hop: int = 128 * 4                                                              # hop size of STFT
    alpha_rescale: float = 0.65                                                     # alpha rescale parameter for STFT representation
    beta_rescale: float = 0.34                                                      # beta rescale parameter for STFT representation
    sigma_data: float = 0.5                                                         # sigma data for EDM framework

    # EVALUATION
    eval_samples_path: str = 'eval_samples'                                         # generated images for FAD evaluation during training will be saved here
    num_samples_fad: int = 1000                                                     # number of samples that are generated for FAD evaluation
    inference_diffusion_steps: int = 1                                              # how many denoising steps to use for FAD calculation
    fad_models: List[str] = field(default_factory=lambda: ['vggish', 'clap'])       # list of FAD models to use
    fad_workers: int = 16                                                           # number of workers for FAD evaluation
    fad_background_embeddings: List[str] = field(default_factory=lambda: [f'fad_stats/test_data_fad_embeddings_{fm}.npy' for fm in ['vggish', 'clap']])    # name of FAD embeddings file. If does not exist, it will be created on the first run

    # MODEL
    base_channels: int = 64                                                         # base channel number for architecture
    layers_list: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])         # number of blocks per each resolution level
    multipliers_list: List[int] = field(default_factory=lambda: [1, 2, 4, 4, 4])    # base channels multipliers for each resolution level
    attention_list: List[int] = field(default_factory=lambda: [0, 0, 1, 1, 1])      # for each resolution, 0 if no attention is performed, 1 if attention is performed
    freq_downsample_list: List[int] = field(default_factory=lambda: [1, 0, 0, 0])   # for each resolution, 0 if frequency 4x downsampling, 1 if standard frequency 2x and time 2x downsampling
    layers_list_encoder: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1]) # number of blocks per each resolution level
    attention_list_encoder: List[int] = field(default_factory=lambda: [0, 0, 1, 1, 1])  # for each resolution, 0 if no attention is performed, 1 if attention is performed
    bottleneck_base_channels: int = 512                                             # base channels to use for block before/after bottleneck
    num_bottleneck_layers: int = 4                                                  # number of blocks to use before/after bottleneck
    frequency_scaling: bool = True                                                  # use frequency scaling
    heads: int = 4                                                                  # number of attention heads
    cond_channels: int = 256                                                        # dimension of time embedding
    use_fourier: bool = False                                                       # if True, use random Fourier embedding, if False, use Positional
    fourier_scale: float = 0.2                                                      # scale parameter for gaussian fourier layer
    normalization: bool = True                                                      # use group normalization
    dropout_rate: float = 0.0                                                       # dropout rate
    min_res_dropout: int = 16                                                       # dropout is applied on equal or smaller feature map resolutions
    init_as_zero: bool = True                                                       # initialize convolution kernels before skip connections with zero-weighted kernels
    bottleneck_channels: int = 64                                                   # channels of encoder bottleneck
    pre_normalize_2d_to_1d: bool = True                                             # pre-normalize 2D to 1D connection in encoder
    pre_normalize_downsampling_encoder: bool = True                                 # pre-normalize downsampling layers in encoder

    # DIFFUSION PARAMETERS
    schedule: str = 'exponential'                                                   # step schedule to use ['constant', 'exponential']
    start_exp: float = 1.0                                                          # if schedule is 'exponential', the starting exponent
    end_exp: float = 3.0                                                            # the higher the exponent, the smaller the steps at the end of training or throughout training if schedule is 'constant'
    base_step: float = 0.1                                                          # the base step on which the exponent is applied
    sigma_min: float = 0.002                                                        # minimum sigma for EDM framework
    sigma_max: float = 80.0                                                         # maximum sigma for EDM framework
    rho: float = 7.0                                                                # rho parameter for EDM framework
    use_lognormal: bool = True                                                      # use a lognormal noise schedule during training
    p_mean: float = -1.1                                                            # mean of lognormal noise schedule
    p_std: float = 2.0                                                              # standard deviation of lognormal noise schedule

    def update(self, config_dict: dict):
        # make sure I can set each key as an attribute (e.g. I can call hparams.batch_size)
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # update data paths
        self.data_paths = config_dict['data_paths']
        self.data_path_test = config_dict['data_path_test']


# Create a global instance of the dataclass
hparams = HParams()