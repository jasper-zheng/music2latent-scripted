#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forward-wav-inference.py
------------------------

• Load waveform from --input (.wav)
• Load scripted UNet from --model
• Feed the waveform through model.forward() N times (CPU / f32)
• Save final audio to --output (default: out.wav)

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torchaudio


def pick_device_and_dtype(ts_module: torch.jit.ScriptModule) -> tuple[torch.device, torch.dtype]:
    try:
        p = next(ts_module.parameters())
        return p.device, p.dtype
    except StopIteration:
        try:
            b = next(ts_module.buffers())
            return b.device, b.dtype
        except StopIteration:
            return torch.device("cpu"), torch.float32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=Path, default="real-cello-arp-test-0-fast-6st.wav",
                   help="input .wav file to run forward() on")
    p.add_argument("--model", type=Path, default="arpitect-v0m-scripted-02072025-CPU.ts",
                   help="TorchScript UNet with .forward()")
    p.add_argument("--output", type=Path, default="arpitect_v0m_fwd-out-cello-arp-test-0-fast-6st.wav",
                   help="destination .wav path")
    p.add_argument("--forward", type=int, default=0, metavar="N",
                   help="how many times to call model.forward()")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load input audio ---------------------------------------------------
    print(f"→ loading input audio: {args.input}")
    waveform, sr = torchaudio.load(str(args.input))  # (C, S)
    waveform = waveform.float().cpu()

    print(f"   waveform shape: {tuple(waveform.shape)}  @ {sr} Hz")

    # --- Load model ---------------------------------------------------------
    print(f"→ loading scripted model: {args.model}")
    model = torch.jit.load(str(args.model)).eval()
    model = model.to(torch.device("cpu"), dtype=torch.float32)

    # --- Prepare for forward ------------------------------------------------
    wave_batched = waveform.unsqueeze(0) if waveform.ndim == 2 else waveform  # (B, C, S)

    # --- Run forward passes -------------------------------------------------
    print(f"→ running {args.forward} forward round-trip(s) on CPU / f32")
    for n in range(1, args.forward + 1):
        t0 = time.time()
        with torch.no_grad():
            wave_batched = model.forward(wave_batched)
        print(f"   pass {n}/{args.forward} done in {time.time() - t0:.3f}s")

    # --- Save final result --------------------------------------------------
    out_wave = wave_batched.squeeze(0).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), out_wave, sr)
    print(f"✓ wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
