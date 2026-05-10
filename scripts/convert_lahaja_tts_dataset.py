#!/usr/bin/env python3
"""
Convert Lahaja TTS dataset to LlamaFactory ShareGPT format with Mimi codec.
"""

import os
import json
import csv
import argparse
from tqdm import tqdm
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Lahaja TTS dataset to ShareGPT format")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--audio_base_path", type=str, required=True, help="Base path for audio files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--delimiter", type=str, default="|", help="CSV delimiter")
    parser.add_argument("--mimi_model", type=str, default="kyutai/mimi", help="Mimi model ID")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--prompt_template", type=str, 
                     default="<|spk_start|>{speaker}<|spk_end|><|audio|>{text}",
                     help="Template for user prompt")
    return parser.parse_args()

def load_csv_data(csv_file, delimiter="|"):
    """Load CSV data with path|speaker|text format."""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if 'path' in row and 'speaker' in row and 'text' in row:
                data.append({
                    'path': row['path'],
                    'speaker': row['speaker'], 
                    'text': row['text']
                })
    return data

def main():
    args = parse_args()
    
    print(f"Loading Lahaja CSV from {args.csv_file}...")
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
    
    for item in tqdm(data, desc="Processing Lahaja audio"):
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
                    {"from": "system", "value": "You are Qwen, a virtual human developed by Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."},
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
        print(f"  Prompt: {sample['conversations'][1]['value']}")
        print(f"  Response: {sample['conversations'][2]['value']}")
        print(f"  Speaker: {sample['speaker']}")
        print(f"  Audio codes: {len(sample['audio_codes'])} codebooks x {len(sample['audio_codes'][0]) if sample['audio_codes'] else 0} tokens")


if __name__ == "__main__":
    main()
