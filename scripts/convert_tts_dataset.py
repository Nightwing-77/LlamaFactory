#!/usr/bin/env python3
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

"""Convert TTS dataset to LlamaFactory format with Mimi audio codec targets.

This script converts audio-text pairs to ShareGPT format with Mimi codec tokens.
The Mimi codec is used ONLY to extract target audio codes - we don't train it!

Usage:
    python convert_tts_dataset.py \
        --dataset_path /path/to/audio/files \
        --metadata_file metadata.csv \
        --output_file tts_dataset.json \
        --text_column text \
        --audio_column audio_path \
        --mimi_model kyutai/mimi

Input format (CSV):
    text,audio_path
    "Hello world","audio1.wav"
    "How are you?","audio2.wav"

Output format (JSON):
    [
        {
            "conversations": [
                {"from": "human", "value": "Generate speech: Hello world"},
                {"from": "gpt", "value": "<|audio|>"}
            ],
            "audio_codes": [[123, 456, ...], [234, 567, ...], ...]  // Mimi codec tokens
        }
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add LlamaFactory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TTS dataset with Mimi codec targets")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to audio files directory")
    parser.add_argument("--metadata_file", type=str, required=True, help="CSV/JSON file with text and audio paths")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text")
    parser.add_argument("--audio_column", type=str, default="audio_path", help="Column name for audio path")
    parser.add_argument("--speaker_column", type=str, default=None, help="Column name for speaker ID (optional)")
    parser.add_argument("--mimi_model", type=str, default="kyutai/mimi", help="Mimi model ID")
    parser.add_argument("--prompt_template", type=str, default="Generate speech: {text}", help="Prompt template")
    parser.add_argument("--use_speaker_tokens", action="store_true", help="Use speaker tokens (Chelsie|Serena)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for Mimi encoding")
    return parser.parse_args()


def load_metadata(metadata_file: str):
    """Load metadata from CSV or JSON file."""
    import pandas as pd
    
    if metadata_file.endswith(".csv"):
        return pd.read_csv(metadata_file)
    elif metadata_file.endswith(".json"):
        return pd.read_json(metadata_file)
    else:
        raise ValueError("Metadata file must be CSV or JSON")


def main():
    args = parse_args()
    
    print(f"Loading metadata from {args.metadata_file}...")
    df = load_metadata(args.metadata_file)
    print(f"Loaded {len(df)} samples")
    
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Using first {len(df)} samples")
    
    # Initialize Mimi codec
    print(f"Loading Mimi codec from {args.mimi_model}...")
    try:
        from src.llamafactory.extras.mimi_codec import MimiCodec
        mimi = MimiCodec(model_id=args.mimi_model)
    except ImportError as e:
        print(f"Error loading Mimi: {e}")
        print("Please install: pip install git+https://github.com/kyutai-labs/momi.git")
        sys.exit(1)
    
    # Process each sample
    output_data = []
    errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        try:
            text = str(row[args.text_column])
            audio_path = os.path.join(args.dataset_path, str(row[args.audio_column]))
            
            # Get speaker ID if available
            speaker_id = None
            if args.speaker_column and args.speaker_column in row:
                speaker_id = str(row[args.speaker_column])
            
            # Check audio file exists
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                errors += 1
                continue
            
            # Encode audio with Mimi
            audio_codes = mimi.encode_file(audio_path)
            
            # Create ShareGPT format entry
            # Voice format: <|spk_start|>{speaker_id}<|spk_end|><|audio|>
            if args.use_speaker_tokens and speaker_id:
                # Qwen2.5-Omni supports: Chelsie, Serena (and others)
                assistant_value = f"<|spk_start|>{speaker_id}<|spk_end|><|audio|>"
            else:
                assistant_value = "<|audio|>"  # Default voice
            
            entry = {
                "conversations": [
                    {"from": "human", "value": args.prompt_template.format(text=text)},
                    {"from": "gpt", "value": assistant_value}
                ],
                "audio_codes": audio_codes  # Mimi codec targets
            }
            
            # Add speaker metadata if available
            if speaker_id:
                entry["speaker"] = speaker_id
            
            output_data.append(entry)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            errors += 1
            continue
    
    # Save output
    print(f"\nSaving {len(output_data)} samples to {args.output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Processed {len(df)} samples, {len(output_data)} successful, {errors} errors")
    
    # Print sample
    if output_data:
        print("\nSample entry:")
        sample = output_data[0]
        print(f"  Prompt: {sample['conversations'][0]['value']}")
        print(f"  Audio codes shape: {len(sample['audio_codes'])} codebooks x {len(sample['audio_codes'][0]) if sample['audio_codes'] else 0} tokens")
        print(f"  First 5 tokens of first codebook: {sample['audio_codes'][0][:5] if sample['audio_codes'] else 'N/A'}")


if __name__ == "__main__":
    main()
