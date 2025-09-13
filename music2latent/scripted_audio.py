# Reused some functions from https://github.com/SonyCSLParis/music2latent/blob/master/music2latent/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, List

class StreamingSTFT(nn.Module):
    """Streaming STFT processor with buffer management"""
    
    def __init__(self, hop_size: int, fac: int, use_power_ceil: bool = False, buffer_size: Optional[int] = None):
        super(StreamingSTFT, self).__init__()
        self.hop_size = hop_size
        self.fac = fac
        self.frame_length = fac * hop_size
        
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
            STFT frames [batch_size, data_channel, freq_bins, time_frames]
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
        # total_length = (num_frames - 1) * self.hop_size + frame_length

        self.register_buffer('output_buffer', torch.zeros(0))  # Start with empty buffer
        window = torch.hann_window(self.frame_length)
        inv_window = inverse_stft_window(window, self.frame_length, self.hop_size)
        self.register_buffer('inv_window', inv_window)
        self.initialized += 1
    
    def process_chunk(self, stft_frames: torch.Tensor) -> torch.Tensor:
        """
        Process STFT frames back to audio with overlap-add
        
        Args:
            stft_frames: STFT frames [batch_size, data_channel, freq_bins, time_frames]
            
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
        # print(f"batch_size: {batch_size}, num_frames: {num_frames}, frame_length: {frame_length}")
        # Calculate total length needed for all frames
        total_length = (num_frames - 1) * self.hop_size + frame_length
        
        # Create output buffer for this processing (including overlap regions)
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

def inverse_stft_window(forward_window, frame_length:int, frame_step:int):
    denom = forward_window**2
    overlaps = -(-frame_length // frame_step)
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = torch.reshape(denom, [overlaps, frame_step])
    denom = torch.sum(denom, 0, keepdim=True)
    denom = torch.tile(denom, [overlaps, 1])
    denom = torch.reshape(denom, [overlaps * frame_step])
    return forward_window / denom[:frame_length]

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
