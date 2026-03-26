from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel

from .stft import STFT


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half: bool,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        mel_fmin: int = 0,
        mel_fmax: int = None,
        clamp: float = 1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(
        self,
        audio: torch.Tensor,
        keyshift: int = 0,
        speed: int = 1,
        center: bool = True,
    ) -> torch.Tensor:
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )
        if "privateuseone" in str(audio.device):
            if not hasattr(self, "stft"):
                self.stft = STFT(
                    filter_length=n_fft_new,
                    hop_length=hop_length_new,
                    win_length=win_length_new,
                    window="hann",
                ).to(audio.device)
            magnitude = self.stft.transform(audio)
        else:
            fft = torch.stft(
                audio,
                n_fft=n_fft_new,
                hop_length=hop_length_new,
                win_length=win_length_new,
                window=self.hann_window[keyshift_key],
                center=center,
                return_complex=True,
            )
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec
