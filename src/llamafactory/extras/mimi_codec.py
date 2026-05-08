# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mimi Codec integration for TTS training.

Extracts audio codec tokens from audio files for training targets.
Uses the Mimi neural audio codec (32 kHz, 32 RVQ codebooks).
"""

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from . import logging
from .packages import is_librosa_available, is_mimi_available


logger = logging.get_logger(__name__)


class MimiCodec:
    """Mimi neural audio codec for extracting audio tokens.
    
    This is used ONLY for extracting target audio codes for TTS training.
    The audio encoder in the model remains FROZEN - we don't train it.
    
    Mimi codec details:
    - Sample rate: 32 kHz
    - RVQ: 32 codebooks, each with 2048 entries
    - Frame rate: 12.5 Hz (80ms per frame)
    - Compression: ~0.5 kbps per codebook, ~16 kbps total
    """
    
    def __init__(self, model_id: str = "kyutai/mimi", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        r"""Initialize Mimi codec.
        
        Args:
            model_id: HuggingFace model ID for Mimi
            device: Device to load model on
        """
        if not is_mimi_available():
            raise ImportError(
                "Mimi codec requires 'mimi' package. "
                "Install with: pip install git+https://github.com/kyutai-labs/mimi.git"
            )
        
        from moshi.models import loaders  # type: ignore
        
        self.device = device
        self.model_id = model_id
        
        # Load Mimi model
        logger.info_rank0(f"Loading Mimi codec from {model_id}...")
        self.model = loaders.get_mimi(model_id, device=device)
        self.model.eval()
        
        # Mimi config
        self.sample_rate = 24000  # Mimi expects 24kHz
        self.num_codebooks = 32  # RVQ with 32 codebooks
        self.vocab_size = 2048  # Each codebook has 2048 entries
        
        logger.info_rank0(f"Mimi codec loaded: {self.num_codebooks} codebooks, vocab_size={self.vocab_size}")
    
    @torch.no_grad()
    def encode(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        r"""Encode audio to codec tokens.
        
        Args:
            audio: Audio waveform as numpy array or tensor [length] or [batch, length]
                   Should be at 24kHz sample rate
        
        Returns:
            Codec tokens tensor of shape [batch, num_codebooks, num_frames]
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape [batch, channels, length]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, length]
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, 1, length]
        
        audio = audio.to(self.device)
        
        # Encode
        codes = self.model.encode(audio)  # [batch, num_codebooks, num_frames]
        
        return codes.cpu()
    
    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        r"""Decode codec tokens back to audio.
        
        Args:
            codes: Codec tokens [batch, num_codebooks, num_frames]
        
        Returns:
            Audio waveform [batch, channels, length]
        """
        codes = codes.to(self.device)
        audio = self.model.decode(codes)
        return audio.cpu()
    
    def encode_file(self, audio_path: str) -> list[list[int]]:
        r"""Encode audio file to codec tokens list format.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            List of codebook token lists: [[cb0_t0, cb0_t1, ...], [cb1_t0, cb1_t1, ...], ...]
        """
        if not is_librosa_available():
            raise ImportError("librosa is required for audio file loading")
        
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Encode
        codes = self.encode(audio)  # [1, num_codebooks, num_frames]
        codes = codes[0]  # Remove batch dim: [num_codebooks, num_frames]
        
        # Convert to list format
        code_list = codes.tolist()  # List of lists
        
        return code_list
    
    def __call__(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convenience method to encode audio."""
        return self.encode(audio)


# Global singleton instance
_mimi_codec: MimiCodec | None = None


def get_mimi_codec(model_id: str = "kyutai/mimi") -> MimiCodec:
    r"""Get or create global Mimi codec instance."""
    global _mimi_codec
    if _mimi_codec is None:
        _mimi_codec = MimiCodec(model_id)
    return _mimi_codec


def reset_mimi_codec() -> None:
    r"""Reset global Mimi codec instance."""
    global _mimi_codec
    _mimi_codec = None
