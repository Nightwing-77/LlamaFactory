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

"""TTS Trainer for audio codec generation training.

This trainer computes loss on audio codec tokens from the talker.codec_head,
not text tokens like standard SFT.
"""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

from ...extras import logging
from ...extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel

logger = logging.get_logger(__name__)


class TTSTrainer(Seq2SeqTrainer):
    r"""Trainer for Text-to-Speech training with audio codec targets.
    
    Key differences from standard SFT:
    1. Computes loss on talker.codec_head outputs, not language model logits
    2. Targets are audio codec tokens, not text tokens
    3. Audio encoder and code2wav are frozen, only training thinker + codec_head
    """
    
    def __init__(self, **kwargs):
        # Disable NEFTune - it doesn't work with Qwen2.5-Omni's architecture
        kwargs.pop('neftune_noise_alpha', None)
        super().__init__(**kwargs)
        self.tts_loss_weight = 1.0
        self.text_loss_weight = 0.0  # We don't train on text for TTS
        
    def compute_loss(self, model: "PreTrainedModel", inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: int | None = None):
        r"""Compute TTS loss on audio codec tokens.
        
        For TTS training:
        1. Forward pass through full model (thinker + talker)
        2. Get codec_head predictions from talker
        3. Compute cross-entropy loss against audio_code_targets
        """
        # Extract audio codec targets if present
        audio_code_targets = inputs.pop("audio_code_targets", None)
        
        # Forward pass through model (PEFT wrapper handles base model dispatch)
        outputs = model.forward(**inputs)
        
        # Get model type
        model_type = getattr(model.config, "model_type", None)
        is_omni_tts = model_type in ["qwen2_5_omni", "qwen2_5_omni_thinker"]
        
        # Compute TTS loss if we have audio codec targets
        if audio_code_targets is not None and is_omni_tts:
            tts_loss = self._compute_tts_loss(model, outputs, audio_code_targets)
            if tts_loss is not None:
                # Replace standard loss with TTS loss
                loss = tts_loss
                if return_outputs:
                    return (loss, outputs)
                return loss
        
        # Fallback to standard loss computation
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        if return_outputs:
            return (loss, outputs)
        return loss
    
    def _compute_tts_loss(
        self, 
        model: "PreTrainedModel", 
        outputs: Any, 
        audio_code_targets: torch.Tensor
    ) -> torch.Tensor | None:
        r"""Compute cross-entropy loss on audio codec tokens.
        
        Args:
            model: The Qwen2.5-Omni model (with talker)
            outputs: Model forward outputs
            audio_code_targets: Target codec tokens [batch, num_codebooks, seq_len]
        
        Returns:
            Scalar loss tensor or None
        """
        # Get hidden states from thinker output (before codec_head)
        # The model should have talker.codec_head that projects to codec space
        if not hasattr(model, "talker") or not hasattr(model.talker, "codec_head"):
            logger.warning_once("Model does not have talker.codec_head, skipping TTS loss")
            return None
        
        # Get thinker hidden states - these are what we train
        # For Qwen2.5-Omni, the thinker generates text/audio representations
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[0]
        
        # Get codec predictions from talker.codec_head
        # codec_head projects hidden states to codec vocabulary
        codec_logits = model.talker.codec_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        # audio_code_targets shape: [batch, num_codebooks, audio_seq_len]
        # We need to align text sequence length with audio sequence length
        batch_size, num_codebooks, audio_seq_len = audio_code_targets.shape
        _, text_seq_len, vocab_size = codec_logits.shape
        
        # Compute loss per codebook
        total_loss = 0.0
        valid_codebooks = 0
        
        for cb_idx in range(num_codebooks):
            # Get targets for this codebook
            cb_targets = audio_code_targets[:, cb_idx, :]  # [batch, audio_seq_len]
            
            # Check if targets are valid (not all IGNORE_INDEX)
            valid_mask = (cb_targets != IGNORE_INDEX)
            if not valid_mask.any():
                continue
            
            # We need to align audio sequence with text sequence
            # This is tricky because audio tokens are at 12.5Hz (80ms/frame)
            # while text tokens are at variable rate
            # For now, use the last hidden states for audio generation
            
            # Truncate to minimum sequence length
            min_len = min(text_seq_len, audio_seq_len)
            cb_logits = codec_logits[:, -min_len:, :]  # Use last positions for audio
            cb_targets_aligned = cb_targets[:, :min_len]
            
            # Reshape for cross entropy: [batch * seq_len, vocab_size]
            cb_logits_flat = cb_logits.reshape(-1, vocab_size)
            cb_targets_flat = cb_targets_aligned.reshape(-1)
            
            # Mask out invalid targets
            valid_mask_flat = (cb_targets_flat != IGNORE_INDEX)
            if not valid_mask_flat.any():
                continue
            
            cb_logits_valid = cb_logits_flat[valid_mask_flat]
            cb_targets_valid = cb_targets_flat[valid_mask_flat]
            
            # Compute loss
            loss = F.cross_entropy(cb_logits_valid, cb_targets_valid)
            total_loss += loss
            valid_codebooks += 1
        
        if valid_codebooks == 0:
            return None
        
        return total_loss / valid_codebooks
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        r"""Override prediction step for TTS evaluation."""
        audio_code_targets = inputs.get("audio_code_targets", None)
        
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # For generation, we would decode codec tokens to audio
        # For now, just return the loss
        return (loss, None, None)
