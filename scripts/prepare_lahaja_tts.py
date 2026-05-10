#!/usr/bin/env python3
"""
Prepare Lahaja dataset for TTS training by creating CSV with path|speaker|text format.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
import csv

def prepare_lahaja_tts(split="test", max_samples=None):
    """Convert Lahaja dataset to TTS format with speaker information."""
    
    # Load Lahaja dataset
    dataset = load_dataset("ai4bharat/Lahaja", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples from Lahaja {split}")
    
    # Create audio directory
    Path("./lahaja_audio").mkdir(exist_ok=True)
    
    # Create CSV data
    csv_data = []
    
    for i, ex in enumerate(dataset):
        text = ex.get("text", "").strip()
        if not text:
            continue
            
        # Extract speaker info from sample ID or use generic
        speaker_id = ex.get("speaker_id", f"speaker_{i % 10}")  # Cycle through 10 speakers
        speaker_name = f"lahaja_{speaker_id}"
        
        # Save audio
        audio_data = ex.get("audio")
        if audio_data is not None:
            audio_array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", 16000)
            
            if audio_array is not None:
                audio_path = f"lahaja_{i:08d}.wav"
                audio_full_path = f"./lahaja_audio/{audio_path}"
                
                # Save audio file
                sf.write(audio_full_path, audio_array, sr)
                
                # Add to CSV
                csv_data.append({
                    'path': audio_path,
                    'speaker': speaker_name,
                    'text': text
                })
                
                if i % 100 == 0:
                    print(f"Processed {i} samples...")
    
    # Save CSV
    os.makedirs("data", exist_ok=True)
    csv_file = f"data/lahaja_tts_{split}.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'speaker', 'text'], delimiter='|')
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Saved {len(csv_data)} samples to {csv_file}")
    
    # Print speaker distribution
    speakers = set(item['speaker'] for item in csv_data)
    print(f"Speakers: {', '.join(sorted(speakers))}")
    
    return csv_file, csv_data

if __name__ == "__main__":
    # Prepare Lahaja TTS dataset
    csv_file, data = prepare_lahaja_tts("test", max_samples=1000)
    print(f"\nCSV file created: {csv_file}")
    print(f"Total samples: {len(data)}")
    
    # Show sample
    if data:
        print(f"\nSample entry:")
        print(f"Path: {data[0]['path']}")
        print(f"Speaker: {data[0]['speaker']}")
        print(f"Text: {data[0]['text'][:100]}...")
