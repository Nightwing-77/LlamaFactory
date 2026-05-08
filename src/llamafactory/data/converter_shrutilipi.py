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
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..extras.packages import is_librosa_available, is_soundfile_available

if TYPE_CHECKING:
    from .parser import DatasetAttr
    from ..hparams import DataArguments

logger = logging.get_logger(__name__)


@dataclass
class ShrutilipiDatasetConverter:
    """Converter for ai4bharat/Shrutilipi dataset.
    
    Dataset format:
    - audio_filepath: bytes (audio data, not filepath!)
    - text: transcription text
    """
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # Get text from 'text' column
        text = example.get("text", "")
        if not text:
            logger.warning("Empty text in example")
            return None
            
        # Get audio bytes from 'audio_filepath' column (misleading name - it's bytes!)
        audio_data = example.get("audio_filepath")
        if audio_data is None:
            logger.warning("No audio data in example")
            return None
            
        # Convert bytes to audio input format
        # AudioInput can be: path string, bytes, or numpy array
        audio_input = self._process_audio_bytes(audio_data)
        if audio_input is None:
            return None
        
        # Create ShareGPT format for STT
        output = {
            "_prompt": "<audio>\nTranscribe this audio",
            "_response": text,
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
            "_audios": [audio_input] if audio_input else None,
        }
        
        return output
    
    def _process_audio_bytes(self, audio_data: bytes) -> Any:
        """Convert audio bytes to format usable by model."""
        try:
            if is_soundfile_available():
                import soundfile as sf
                # Load from bytes
                audio_buffer = io.BytesIO(audio_data)
                waveform, sample_rate = sf.read(audio_buffer)
                return {
                    "array": waveform,
                    "sampling_rate": sample_rate,
                    "path": None,
                }
            elif is_librosa_available():
                import librosa
                audio_buffer = io.BytesIO(audio_data)
                waveform, sample_rate = librosa.load(audio_buffer, sr=None)
                return {
                    "array": waveform,
                    "sampling_rate": sample_rate,
                    "path": None,
                }
            else:
                # Fallback: return bytes directly
                # The mm_plugin should handle this
                return audio_data
        except Exception as e:
            logger.warning(f"Failed to process audio bytes: {e}")
            return None
