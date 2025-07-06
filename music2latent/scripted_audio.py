import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torchaudio
# import librosa
# import matplotlib.pyplot as plt

from typing import Optional, List

class StreamingSTFT(nn.Module):
    """Streaming STFT processor with buffer management"""
    
    def __init__(self, hop_size: int, fac: int):
        super(StreamingSTFT, self).__init__()
        self.hop_size = hop_size
        self.fac = fac
        self.frame_length = fac * hop_size
        # self.device = device
        
        # Input buffer to maintain overlap between chunks
        # self.input_buffer = torch.zeros(self.frame_length - self.hop_size, device=device)
        # self.register_buffer('input_buffer', torch.zeros(self.frame_length - self.hop_size))
        
        # Window function
        # self.window = torch.hann_window(self.frame_length, device=device)
        # self.register_buffer('window', torch.hann_window(self.frame_length))
        self.initialized = 0
        self.init_buffer()
    
    # @torch.jit.unused
    def init_buffer(self):
        self.register_buffer('input_buffer', torch.zeros(self.frame_length - self.hop_size))
        self.register_buffer('window', torch.hann_window(self.frame_length))
        self.initialized += 1

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Process a chunk of audio with STFT
        
        Args:
            chunk: Input audio chunk [batch_size, samples]
            
        Returns:
            STFT frames [batch_size, freq_bins, time_frames]
        """
        # if not self.initialized:
        #     self.init_buffer(chunk)

        batch_size = chunk.shape[0]
        
        # Expand buffer to match batch size if needed
        if self.input_buffer.shape[0] != batch_size:
            self.input_buffer = self.input_buffer.unsqueeze(0).expand(batch_size, -1)
        
        # Prepend buffer to chunk
        padded_chunk = torch.cat([self.input_buffer, chunk], dim=-1)
        
        # Update buffer with end of current chunk
        buffer_size = self.frame_length - self.hop_size
        if chunk.shape[-1] >= buffer_size:
            self.input_buffer = chunk[..., -buffer_size:].clone()
        else:
            # If chunk is smaller than buffer, slide the buffer
            self.input_buffer = torch.cat([
                self.input_buffer[..., chunk.shape[-1]:], 
                chunk
            ], dim=-1)
        
        # Perform STFT on padded chunk
        padded_chunk = self.stft(padded_chunk)
        padded_chunk = padded_chunk[:,:self.hop_size*2,:]
        x = normalize_complex(padded_chunk) 
        return torch.stack((torch.real(x),torch.imag(x)), -3)
    
    def stft(self, wv: torch.Tensor) -> torch.Tensor:
        """STFT implementation"""
        framed_signals = frame(wv, self.frame_length, self.hop_size)
        framed_signals = framed_signals * self.window
        return torch.fft.rfft(framed_signals, n=None, dim=-1, norm=None).permute(0, 2, 1)
    
    def reset(self):
        """Reset the buffer state"""
        self.input_buffer.zero_()


class StreamingISTFT(nn.Module):
    """Streaming ISTFT processor with buffer management"""
    
    def __init__(self, hop_size: int, fac: int):
        super(StreamingISTFT, self).__init__()
        self.hop_size = hop_size
        self.fac = fac
        self.frame_length = fac * hop_size
        
        self.initialized = 0
        self.init_buffer()

    # @torch.jit.unused
    def init_buffer(self):
        self.register_buffer('output_buffer', torch.zeros(0))  # Start with empty buffer
        window = torch.hann_window(self.frame_length)
        inv_window = inverse_stft_window(window, self.frame_length, self.hop_size)
        self.register_buffer('inv_window', inv_window)
        self.initialized += 1
    
    def process_chunk(self, stft_frames: torch.Tensor) -> torch.Tensor:
        """
        Process STFT frames back to audio with overlap-add
        
        Args:
            stft_frames: STFT frames [batch_size, freq_bins, time_frames]
            
        Returns:
            Audio chunk [batch_size, samples]
        """
        
        batch_size = stft_frames.shape[0]
        
        stft_frames = torch.nn.functional.pad(stft_frames, (0,0,0,1))
        real,imag = torch.chunk(stft_frames, 2, -3)
        x = torch.complex(real.squeeze(-3),imag.squeeze(-3))
        x = denormalize_complex(x)
        
        # Expand buffer to match batch size if needed
        if self.output_buffer.numel() > 0 and self.output_buffer.shape[0] != batch_size:
            self.output_buffer = self.output_buffer.unsqueeze(0).expand(batch_size, -1).contiguous()
        
        # Inverse FFT
        x = torch.fft.irfft(x, dim=-2).permute(0,2,1)  # [batch, time_frames, frame_length]
        
        # Apply inverse window
        x = x * self.inv_window.unsqueeze(0).unsqueeze(0) # [batch, time_frames, frame_length]

        # Perform streaming overlap-add
        output_chunk = self.streaming_overlap_add(x)
        
        return output_chunk
    
    def streaming_overlap_add(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Streaming overlap-add that maintains buffer state
        Uses a simple and direct approach similar to the non-streaming version
        
        Args:
            frames: Time-domain frames [batch_size, time_frames, frame_length]
            
        Returns:
            Output audio chunk [batch_size, samples]
        """
        batch_size, num_frames, frame_length = frames.shape
        
        # Calculate total length needed for all frames
        total_length = (num_frames - 1) * self.hop_size + frame_length
        
        # Create output buffer for this processing (including overlap regions)
        # TODO: Buffer this:
        full_output = torch.zeros(batch_size, total_length, device=frames.device)
        
        # Add each frame at its correct position
        for i in range(num_frames):
            frame = frames[:, i, :]
            start_pos = i * self.hop_size
            end_pos = start_pos + frame_length
            full_output[:, start_pos:end_pos] += frame
        
        # Handle the carryover from previous chunk
        if self.output_buffer.numel() > 0:
            buffer_len = self.output_buffer.shape[-1]
            overlap_len = min(buffer_len, total_length)
            if overlap_len > 0:
                full_output[:, :overlap_len] += self.output_buffer[:, :overlap_len]
        
        # Extract the chunk we want to output (hop_size * num_frames)
        chunk_length = self.hop_size * num_frames
        output_chunk = full_output[:, :chunk_length]
        
        # Store the remaining part as buffer for next chunk
        if total_length > chunk_length:
            self.output_buffer = full_output[:, chunk_length:].clone()
        else:
            self.output_buffer = torch.zeros(batch_size, 0, device=frames.device)
        
        return output_chunk
    
    def reset(self):
        """Reset the buffer state"""
        self.output_buffer = torch.zeros(0, device=self.output_buffer.device, dtype=self.output_buffer.dtype)


# Convenience functions for streaming processing
def create_streaming_processors(hop_size: int, fac: int):
    """Create streaming STFT and ISTFT processors"""
    stft_processor = StreamingSTFT(hop_size, fac)
    istft_processor = StreamingISTFT(hop_size, fac)
    return stft_processor, istft_processor


def denormalize_realimag(x, alpha_rescale: float = 0.65, beta_rescale: float = 0.06):
    x = x/beta_rescale
    return torch.sign(x)*(x.abs()**(1./alpha_rescale))

def normalize_complex(x, alpha_rescale: float = 0.65, beta_rescale: float = 0.06):
    return beta_rescale*(x.abs()**alpha_rescale).to(torch.complex64)*torch.exp(1j*torch.atan2(x.imag, x.real).to(torch.complex64))

def denormalize_complex(x, alpha_rescale: float = 0.65, beta_rescale: float = 0.06):
    x = x/beta_rescale
    return (x.abs()**(1./alpha_rescale)).to(torch.complex64)*torch.exp(1j*torch.atan2(x.imag, x.real).to(torch.complex64))

def wv2complex(wv, hop_size: int, fac: int):
    X = stft(wv, hop_size, fac)
    return X[:,:hop_size*2,:]

def wv2realimag(wv, hop_size: int, fac: int):
    X = wv2complex(wv, hop_size, fac)
    X = normalize_complex(X)
    return torch.stack((torch.real(X),torch.imag(X)), -3)

def realimag2wv(x, hop_size: int, fac: int):
    x = torch.nn.functional.pad(x, (0,0,0,1))
    real,imag = torch.chunk(x, 2, -3)
    X = torch.complex(real.squeeze(-3),imag.squeeze(-3))
    X = denormalize_complex(X)
    return istft(X, fac=fac, hop_size=hop_size).clamp(-1.,1.)

# def to_representation_encoder(x, hop_size: int = 512):
#     return wv2realimag(x, hop_size)

# def to_representation(x, hop_size: int = 512):
#     return wv2realimag(x, hop_size)

# def to_waveform(x, hop_size: int = 512):
#     return realimag2wv(x, hop_size)

# Streaming versions of audio conversion functions
def streaming_wv2realimag(streaming_stft: StreamingSTFT, chunk: torch.Tensor) -> torch.Tensor:
    """Convert audio chunk to real/imaginary representation using streaming STFT"""
    X = streaming_stft.process_chunk(chunk)
    X = X[:, :streaming_stft.hop_size*2, :]  # Limit frequency bins
    X_complex = X.permute(0, 2, 1)  # [batch, time, freq]
    X_normalized = normalize_complex(X_complex)
    return torch.stack((torch.real(X_normalized), torch.imag(X_normalized)), -3)

def streaming_realimag2wv(streaming_istft: StreamingISTFT, x: torch.Tensor) -> torch.Tensor:
    """Convert real/imaginary representation back to audio using streaming ISTFT"""
    # The input x is already in the correct format for the streaming ISTFT
    # Just need to permute dimensions to match expected input format
    return streaming_istft.process_chunk(x).clamp(-1., 1.)

def streaming_to_representation(streaming_stft: StreamingSTFT, chunk: torch.Tensor, hop_size: int = 512) -> torch.Tensor:
    """Streaming version of to_representation"""
    return streaming_wv2realimag(streaming_stft, chunk)

def streaming_to_waveform(streaming_istft: StreamingISTFT, x: torch.Tensor, hop_size: int = 512) -> torch.Tensor:
    """Streaming version of to_waveform"""
    return streaming_realimag2wv(streaming_istft, x)

def full_shape(inner_shape: List[int], outer_dimensions: List[int]):
    s = torch.cat([torch.tensor(outer_dimensions), torch.tensor(inner_shape)], 0)
    s = list(s)
    s = [int(el) for el in s]
    return s

def overlap_and_add(signal: torch.Tensor, frame_step: int):

    outer_dimensions = signal.shape[:-2]
    outer_rank = torch.numel(torch.tensor(outer_dimensions))

    frame_length = signal.shape[-1]
    frames = signal.shape[-2]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)  # Divide and round up.

    signal = torch.nn.functional.pad(signal, (0, segments * frame_step - frame_length, 0, segments))

    shape = full_shape([frames + segments, segments, frame_step], outer_dimensions)
    signal = torch.reshape(signal, shape)

    perm = torch.cat([torch.arange(0, outer_rank), torch.tensor([el+outer_rank for el in [1, 0, 2]])], 0)
    perm = list(perm)
    perm = [int(el) for el in perm]
    signal = torch.permute(signal, perm)

    shape = full_shape([(frames + segments) * segments, frame_step], outer_dimensions)
    signal = torch.reshape(signal, shape)

    signal = signal[..., :(frames + segments - 1) * segments, :]

    shape = full_shape([segments, (frames + segments - 1), frame_step], outer_dimensions)
    signal = torch.reshape(signal, shape)

    signal = signal.sum(-3)

    # Flatten the array.
    shape = full_shape([(frames + segments - 1) * frame_step], outer_dimensions)
    signal = torch.reshape(signal, shape)

    # Truncate to final length.
    signal = signal[..., :output_length]

    return signal

def inverse_stft_window(forward_window, frame_length:int, frame_step:int):
    denom = forward_window**2
    overlaps = -(-frame_length // frame_step)
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = torch.reshape(denom, [overlaps, frame_step])
    denom = torch.sum(denom, 0, keepdim=True)
    denom = torch.tile(denom, [overlaps, 1])
    denom = torch.reshape(denom, [overlaps * frame_step])
    return forward_window / denom[:frame_length]

def istft(SP, fac: int, hop_size: int):
    x = torch.fft.irfft(SP, dim=-2)
    window = torch.hann_window(fac*hop_size).to(SP.device)
    window = inverse_stft_window(window, fac*hop_size, hop_size)
    x = x*window.unsqueeze(-1)
    return overlap_and_add(x.permute(0,2,1), hop_size)

# def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
#     """
#     equivalent of tf.signal.frame
#     """
#     signal_length = signal.shape[axis]
#     if pad_end:
#         frames_overlap = frame_length - frame_step
#         rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
#         pad_size = int(frame_length - rest_samples)
#         if pad_size != 0:
#             pad_axis = [0] * signal.ndim
#             pad_axis[axis] = pad_size
#             signal = F.pad(signal, pad_axis, "constant", pad_value)
#     frames = signal.unfold(axis, frame_length, frame_step)
#     return frames
def frame(signal, frame_length: int, frame_step: int):
    """
    equivalent of tf.signal.frame
    """
    # Handle case where signal is shorter than frame_length
    if signal.shape[-1] < frame_length:
        # Pad signal to at least frame_length
        pad_size = frame_length - signal.shape[-1]
        signal = torch.nn.functional.pad(signal, (0, pad_size))
    
    frames = signal.unfold(-1, frame_length, frame_step)
    return frames

def stft(wv, hop_size: int, fac: int):
    window = torch.hann_window(fac*hop_size).to(wv.device)
    framed_signals = frame(wv, fac*hop_size, hop_size)
    framed_signals = framed_signals*window
    return torch.fft.rfft(framed_signals, n=None, dim=- 1, norm=None).permute(0,2,1)

def normalize(S, mu_rescale=-25., sigma_rescale=75.):
    return (S - mu_rescale) / sigma_rescale

def denormalize(S, mu_rescale=-25., sigma_rescale=75.):
    return (S * sigma_rescale) + mu_rescale

def db2power(S_db, ref=1.0):
    return ref * torch.pow(10.0, 0.1 * S_db)

def power2db(power, ref_value=1.0, amin=1e-10):
    log_spec = 10.0 * torch.log10(torch.maximum(torch.tensor(amin), power))
    log_spec -= 10.0 * torch.log10(torch.maximum(torch.tensor(amin), torch.tensor(ref_value)))
    return log_spec