# Music2Latent

Encode and decode audio samples to/from compressed representations! Useful for efficient generative modeling applications and for other downstream tasks.

![music2latent](music2latent.png)

Read the ISMIR 2024 paper [here](https://arxiv.org/abs/2408.06500).
Listen to audio samples [here](https://sonycslparis.github.io/music2latent-companion/).

Under the hood, __Music2Latent__ uses a __Consistency Autoencoder__ model to efficiently encode and decode audio samples.

44.1 kHz audio is encoded into a sequence of __~10 Hz__, and each of the latents has 64 channels.
48 kHz audio can also be encoded, which results in a sequence of ~12 Hz.
A generative model can then be trained on these embeddings, or they can be used for other downstream tasks.

Music2Latent was trained on __music__ and on __speech__. Refer to the [paper](https://arxiv.org/abs/2408.06500) for more details.

## Installation

```bash
pip install music2latent
```

The model weights will be downloaded automatically the first time the code is run for inference.

## How to Use (Inference)

To encode and decode audio samples to/from latent embeddings:

```python
import librosa
from music2latent import EncoderDecoder

audio_path = librosa.example('trumpet')
wv, sr = librosa.load(audio_path, sr=44100)  # Music2Latent supports 48kHz audio as well

encdec = EncoderDecoder()

latent = encdec.encode(wv)
# latent has shape (batch_size/audio_channels, dim (64), sequence_length)

wv_rec = encdec.decode(latent)

# Listen to the reconstructed audio
# import IPython
# IPython.display.display(IPython.display.Audio(wv_rec.squeeze(), rate=sr))
```

To extract encoder features (before the bottleneck) for downstream tasks:

```python
features = encdec.encode(wv, extract_features=True)
# 'features' will have more channels than 'latent' but cannot be decoded.
```
**Loading Custom Trained Models**

The `EncoderDecoder` class, by default, loads our pre-trained model. If you want to use a model you trained yourself, specify the path of the checkpoint of your model in `hparams_inference.py` by changing the `load_path_inference_default` variable.

music2latent supports more advanced usage, including GPU memory management controls. Please refer to __tutorial.ipynb__.

## Training

Make sure your environment is set up with the dependencies listed in `requirements.txt`.
Music2Latent relies on `numpy`, `soundfile`, `huggingface_hub`, `torch>=2.5.0`, `laion-clap`, `torchaudio`, `librosa`, `scipy`.

### 1. Configuration

Music2Latent uses a Python-based configuration system.  Instead of separate `.yaml` or `.json` files, you create a Python file (e.g., `config.py`) that *overrides* default settings.

**Default Hyperparameters:** All the default hyperparameters are defined in `music2latent/hparams.py`.  You *don't* need to copy all of these into your configuration file; you can only specify the ones you want to change.

**Example Configuration File (`config.py`):**

```python
# config.py (example)

batch_size = 16                                                             # batch size
lr = 0.0001                                                                 # learning rate
total_iters = 800000                                                        # total iterations

data_paths = ['/media/datasets/dataset1', '/media/datasets/dataset2']       # list of paths of training datasets (use a single-element list for a single dataset). Audio files will be recursively searched in these paths and in their sub-paths
data_path_test = '/media/datasets/test_dataset'                             # path of samples used for FAD testing (e.g. musiccaps)
```

You always need to specify `data_paths` and `data_path_test`.  `data_paths` should be a *list* of paths to your training datasets.  `data_path_test` should be the path to your test dataset, used for calculating the Frechet Audio Distance (FAD) metric during training.

**Important Hyperparameters:**

*   **`batch_size`:**  Batch size for training
*   **`lr`:**  Initial learning rate.
*   **`lr_decay`:** Learning rate decay schedule (`cosine`, `linear`, `inverse_sqrt`, or `None`).
*   **`total_iters`:** Total number of training iterations.
*   **`data_paths`:** A *list* of paths to your training datasets. The code recursively searches for `.wav` and `.flac` files (or other extensions you specify in `data_extensions`).
*   **`data_fractions`:**  A *list* of sampling weights, specifying how often to sample from each dataset in `data_paths`.  If `None`, datasets are sampled uniformly.
*   **`data_path_test`:**  The path to your *test* dataset, used for calculating the Frechet Audio Distance (FAD) during training.
*   **`compile_model`:**  Whether to use `torch.compile` for potential speedups (see below).
*   **`multi_gpu`:**  Enable multi-GPU training with `torchrun`.
*   **`accumulate_gradients`:**  Accumulates gradients over multiple batches before updating.  This lets you use larger effective batch sizes without exceeding GPU memory.
*   **`checkpoint_path`:**  Directory where checkpoints (saved models) are stored.
*    **`load_path`**:  Load checkpoint from this path to resume training.
*   **`num_workers`**: Number of workers the dataloader will use.

See `music2latent/hparams.py` for *all* available hyperparameters and their default values. You can override *any* of these in your `config.py`.

Also, see the `configs/config.py` file for an example configuration file containing all the hyperparameters and a description of each. You can copy this file and modify it to suit your needs.

### 2. Launching a Training Run

To start a training run, use the `launch.py` script with the `--config` argument:

```bash
python launch.py --config path/to/your/config.py
```

**Checkpoints:**  During training, checkpoints will be stored in the directory specified by the `checkpoint_path` hyperparameter (default: `checkpoints`).  The best checkpoint (lowest FAD) and/or the latest checkpoint will be kept during training.

**TensorBoard:** Training progress (loss, FAD, audio samples, etc.) is logged using TensorBoard.  You can view these logs by running:

```bash
tensorboard --logdir=<your_checkpoint_path>
```

Replace `<your_checkpoint_path>` with the actual path to your checkpoint directory (by default this is `checkpoints`).

### 3. Multi-GPU Training

To use multiple GPUs, use `torchrun`.

**Example (using 3 GPUs):**

```bash
CONDA_VISIBLE_DEVICES=0,1,2 torchrun --nnodes=1 --nproc_per_node=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 launch.py --config my_config.py
```

*   **`CONDA_VISIBLE_DEVICES=0,1,2`:**  This makes GPUs 0, 1, and 2 visible to your training process.  Adjust this based on your system's GPU configuration.
*   **`--nnodes=1`:**  We're running on a single machine (node).
*   **`--nproc_per_node=3`:**  We're using 3 GPUs (processes per node).
*   **`--rdzv_backend=c10d`:**  Specifies the rendezvous backend (how the processes find each other).
*   **`--rdzv_endpoint=localhost:0`:** Specifies the rendezvous endpoint (use a free port; `0` will often pick a random free port).
*   **`launch.py`:**  Our training script.
*   **`--config my_config.py`:** Your configuration file.  **Make sure to set `multi_gpu = True` in your `config.py`.**

**Important:**  When using `torchrun`, each GPU will process a batch size that is equal to `batch_size` divided by the number of GPUs.  For example, if `batch_size = 16` and you're using 3 GPUs, each GPU will process a batch size of 5 (16 / 3 = 5.33, rounded down to 5).

### 4. Model Compilation (`torch.compile`)

Music2Latent supports `torch.compile`, a feature introduced in PyTorch 2.0 that can significantly speed up training.

*   **Enabling Compilation:** Set `compile_model = True` in your configuration file (it's `True` by default).
*   **First Run:** The *first* time you run with `compile_model = True`, PyTorch will compile your model.  This can take a *significant* amount of time (e.g., 10+ minutes, possibly longer, depending on your hardware).  The compiled model will be cached, so subsequent runs will be much faster.
*   **Cache Directory:** The compiled model is cached in the directory specified by `torch_compile_cache_dir` (default: `tmp/torch_compile`).

### 5. Resuming Training

To resume training from a checkpoint, set the `load_path` parameter in your `config.py` to the path of the checkpoint file you want to resume from.  Also, consider setting `load_optimizer = False` if you encounter issues resuming.

```python
# config.py (for resuming)
load_path = "checkpoints/my_run/model_fid_X_loss_X_iters_X.pt"
```

## License

This library is released under the CC BY-NC 4.0 license. Please refer to the LICENSE file for more details.

This work was conducted by [Marco Pasini](https://twitter.com/marco_ppasini) during his PhD at Queen Mary University of London, in partnership with Sony Computer Science Laboratories Paris.
This work was supervised by Stefan Lattner and George Fazekas.
