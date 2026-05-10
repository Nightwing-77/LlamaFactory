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

"""Convert Hindi TTS dataset to LlamaFactory format with Mimi audio codec targets.

This script converts audio-text pairs in pipe-separated CSV format to ShareGPT format.

Input format (CSV with pipe separator):
    path|speaker|text
    kavya/00002.wav|kavya|बता दें कि ये फोटो...
    agastya/00002.wav|agastya|बता दें कि ये फोटो...

Output format (JSON):
    [
        {
            "conversations": [
                {"from": "human", "value": "Generate speech in kavya's voice: बता दें कि ये फोटो..."},
                {"from": "gpt", "value": "<|spk_start|>kavya<|spk_end|><|audio|>"}
            ],
            "audio_codes": [[123, 456, ...], [234, 567, ...], ...],
            "speaker": "kavya"
        }
    ]

Usage:
    python scripts/convert_hindi_tts_dataset.py \
        --csv_file dataset.csv \
        --audio_base_path /path/to/audio/root \
        --output_file hindi_tts_dataset.json \
        --mimi_model kyutai/mimi
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Hindi TTS dataset with Mimi codec targets")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to pipe-separated CSV file")
    parser.add_argument("--audio_base_path", type=str, required=True, help="Base path to audio files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--mimi_model", type=str, default="kyutai/mimi", help="Mimi model ID")
    parser.add_argument("--prompt_template", type=str, default="Generate speech in {speaker}'s voice: {text}", help="Prompt template")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--delimiter", type=str, default="|", help="CSV delimiter (default: |)")
    return parser.parse_args()


def load_csv_data(csv_file: str, delimiter: str = "|"):
    """Load CSV with custom delimiter."""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 3:
                data.append({
                    'path': row[0].strip(),
                    'speaker': row[1].strip(),
                    'text': row[2].strip()
                })
    return data


def main():
    args = parse_args()
    
    print(f"Loading CSV from {args.csv_file}...")
    data = load_csv_data(args.csv_file, args.delimiter)
    print(f"Loaded {len(data)} samples")
    
    # Get unique speakers
    speakers = set(d['speaker'] for d in data)
    print(f"Found speakers: {', '.join(sorted(speakers))}")
    
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Using first {len(data)} samples")
    
    # Initialize Mimi codec
    print(f"Loading Mimi codec from {args.mimi_model}...")
    try:
        from src.llamafactory.extras.mimi_codec import MimiCodec
        mimi = MimiCodec(model_id=args.mimi_model)
    except ImportError as e:
        print(f"Error loading Mimi: {e}")
        print("Please install: pip install git+https://github.com/kyutai-labs/mimi.git")
        sys.exit(1)
    
    # Process each sample
    output_data = []
    errors = 0
    
    for item in tqdm(data, desc="Processing audio"):
        try:
            audio_path = os.path.join(args.audio_base_path, item['path'])
            speaker = item['speaker']
            text = item['text']
            
            # Check audio file exists
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                errors += 1
                continue
            
            # Encode audio with Mimi
            audio_codes = mimi.encode_file(audio_path)
            
            # Create ShareGPT format entry
            entry = {
                "conversations": [
                    {"from": "human", "value": args.prompt_template.format(speaker=speaker, text=text)},
                    {"from": "gpt", "value": f"<|spk_start|>{speaker}<|spk_end|><|audio|>"}
                ],
                "audio_codes": audio_codes,
                "speaker": speaker
            }
            
            output_data.append(entry)
            
        except Exception as e:
            print(f"Error processing {item.get('path', 'unknown')}: {e}")
            errors += 1
            continue
    
    # Save output
    print(f"\nSaving {len(output_data)} samples to {args.output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Processed {len(data)} samples, {len(output_data)} successful, {errors} errors")
    
    # Print sample
    if output_data:
        print("\nSample entry:")
        sample = output_data[0]
        print(f"  Prompt: {sample['conversations'][0]['value']}")
        print(f"  Response: {sample['conversations'][1]['value']}")
        print(f"  Speaker: {sample['speaker']}")
        print(f"  Audio codes: {len(sample['audio_codes'])} codebooks x {len(sample['audio_codes'][0]) if sample['audio_codes'] else 0} tokens")


if __name__ == "__main__":
    main()
