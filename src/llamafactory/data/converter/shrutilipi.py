"""
Shrutilipi dataset converter for direct streaming.
Maps audio bytes from 'audio_filepath' and text from 'text' columns.
"""
from typing import Any
from ..utils import get_logger

logger = get_logger(__name__)


def convert_shrutilipi_example(example: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Shrutilipi example to LlamaFactory ShareGPT format.
    
    Shrutilipi columns:
    - audio_filepath: dict with 'bytes' containing audio data
    - text: transcription text
    
    Output format:
    {
        "conversations": [
            {"from": "human", "value": "<audio>\nTranscribe this audio"},
            {"from": "gpt", "value": text}
        ],
        "audios": [audio_dict]  # Pass bytes dict directly
    }
    """
    audio_data = example.get("audio_filepath")
    text = example.get("text", "")
    
    if audio_data is None:
        raise ValueError("Missing 'audio_filepath' in example")
    
    # audio_data is already in HF datasets format: {"bytes": ..., "path": ...}
    # The mm_plugin will handle the bytes
    return {
        "conversations": [
            {"from": "human", "value": "<audio>\nTranscribe this audio"},
            {"from": "gpt", "value": text}
        ],
        "audios": [audio_data]  # Pass the dict directly, not a path
    }


def shrutilipi_formatter(example: dict[str, Any]) -> dict[str, Any]:
    """Entry point for dataset formatting."""
    return convert_shrutilipi_example(example)
