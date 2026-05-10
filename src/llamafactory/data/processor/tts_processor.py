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

"""TTS (Text-to-Speech) Dataset Processor for audio codec generation training.

This processor handles datasets where:
- Input: Text prompts (e.g., "Say this in Hindi: Hello")
- Target: Audio codec tokens (extracted from target audio using Mimi or similar)

The audio codec tokens are used as training targets instead of text tokens.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen
from .supervised import SupervisedDatasetProcessor

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class TTSDatasetProcessor(SupervisedDatasetProcessor):
    """Processor for TTS training with audio codec token targets.
    
    Extends SupervisedDatasetProcessor to handle audio codec tokens as targets.
    """

    def _encode_tts_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        audio_codes: Optional[list[list[int]]] = None,  # Target audio codec tokens
    ) -> tuple[list[int], list[int], Optional[list[list[int]]]]:
        """Encode a TTS example with audio codec targets.
        
        Returns:
            input_ids: Input token IDs
            labels: Labels for loss computation (using IGNORE_INDEX for non-targets)
            audio_code_targets: Target audio codec tokens for TTS loss
        """
        # Process messages with multimodal plugin
        messages = self.template.mm_plugin.process_messages(
            prompt + response, images, videos, audios, self.processor
        )
        
        # Get input token IDs
        input_ids, _ = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        
        # Encode multiturn conversation
        discarding_history_cot = self.data_args.mask_history and not self.template.preserve_thinking
        encoded_pairs = self.template.encode_multiturn(
            self.tokenizer, messages, system, tools, discarding_history_cot
        )
        
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]

        final_input_ids = []
        final_labels = []
        
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            # For TTS, mask the prompt (source) with IGNORE_INDEX
            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            # For TTS, mask text targets with IGNORE_INDEX (we'll use audio_code_targets instead)
            if self.data_args.mask_history and turn_idx != 0:
                target_label = [IGNORE_INDEX] * target_len
            else:
                # Mask text response - we'll use audio codes as actual targets
                target_label = [IGNORE_INDEX] * target_len

            if self.data_args.mask_history:
                final_input_ids = source_ids + target_ids + final_input_ids
                final_labels = source_label + target_label + final_labels
            else:
                final_input_ids += source_ids + target_ids
                final_labels += source_label + target_label

        if self.template.efficient_eos:
            final_input_ids += [self.tokenizer.eos_token_id]
            final_labels += [self.tokenizer.eos_token_id]

        return final_input_ids, final_labels, audio_codes

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Preprocess TTS dataset with audio codec targets."""
        model_inputs = defaultdict(list)
        
        # Check if audio_codes column exists
        has_audio_codes = "_audio_codes" in examples or "audio_codes" in examples
        
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid TTS example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # Get audio codec targets if available
            audio_codes = None
            if has_audio_codes:
                # Try _audio_codes first (from aligned dataset with underscore prefix)
                if "_audio_codes" in examples and i < len(examples["_audio_codes"]):
                    audio_codes = examples["_audio_codes"][i]
                # Fall back to audio_codes (without underscore)
                elif "audio_codes" in examples and i < len(examples["audio_codes"]):
                    audio_codes = examples["audio_codes"][i]

            input_ids, labels, audio_code_targets = self._encode_tts_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                audio_codes=audio_codes,
            )
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            
            # Add audio codec targets for TTS training
            if audio_code_targets is not None:
                model_inputs["audio_code_targets"].append(audio_code_targets)
            else:
                model_inputs["audio_code_targets"].append([])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Print TTS data example with audio codec targets."""
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
        
        if "audio_code_targets" in example:
            print(f"audio_code_targets:\n{example['audio_code_targets'][:5]}...")  # Show first 5
