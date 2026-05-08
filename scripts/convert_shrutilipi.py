#!/usr/bin/env python3
"""
Convert ai4bharat/Shrutilipi dataset to LlamaFactory format.
Handles audio bytes in 'audio_filepath' column (misleading name - contains bytes!)
"""
import argparse
import io
import json
import os
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def convert_shrutilipi(
    subset: str = "hindi",
    output_dir: str = "data/shrutilipi",
    max_samples: int = None,
    save_audio: bool = True,
    streaming: bool = False
):
    """Convert Shrutilipi dataset to LlamaFactory format."""
    
    print(f"Loading Shrutilipi dataset (subset: {subset}, streaming: {streaming})...")
    
    # Load dataset
    ds = load_dataset(
        "ai4bharat/Shrutilipi",
        subset,
        split="train",
        streaming=streaming,
        trust_remote_code=True
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_audio:
        (output_path / "audio").mkdir(exist_ok=True)
    
    converted = []
    audio_count = 0
    
    # Process samples
    iterator = ds if streaming else tqdm(ds, desc="Converting")
    
    for idx, sample in enumerate(iterator):
        if max_samples and idx >= max_samples:
            break
            
        try:
            # Get text transcription
            text = sample.get("text", "").strip()
            if not text:
                continue
            
            # Get audio bytes from 'audio_filepath' column (contains bytes, not path!)
            audio_data = sample.get("audio_filepath")
            if audio_data is None:
                continue
            
            audio_path = None
            if save_audio:
                # Save audio to file
                audio_filename = f"audio_{idx:08d}.wav"
                audio_path_full = output_path / "audio" / audio_filename
                
                # Load from bytes and save
                audio_buffer = io.BytesIO(audio_data)
                waveform, sample_rate = sf.read(audio_buffer)
                sf.write(audio_path_full, waveform, sample_rate)
                
                audio_path = f"audio/{audio_filename}"
                audio_count += 1
            
            # Create LlamaFactory format
            entry = {
                "conversations": [
                    {"from": "human", "value": "<audio>\nTranscribe this audio"},
                    {"from": "gpt", "value": text}
                ],
                "audios": [audio_path] if audio_path else []
            }
            converted.append(entry)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save JSON
    output_file = output_path / f"{subset}_train.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Samples: {len(converted)}")
    print(f"   Audio files: {audio_count}")
    print(f"   Output: {output_file}")
    
    # Print dataset_info.json snippet
    print(f"\n📋 Add this to dataset_info.json:")
    print(f"""
  "shrutilipi_{subset}": {{
    "file_name": "{subset}_train.json",
    "formatting": "sharegpt",
    "columns": {{
      "messages": "conversations",
      "audios": "audios"
    }},
    "tags": {{
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }}
  }}
""")


def main():
    parser = argparse.ArgumentParser(description="Convert Shrutilipi dataset")
    parser.add_argument("--subset", type=str, default="hindi", help="Dataset subset")
    parser.add_argument("--output_dir", type=str, default="data/shrutilipi", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--no_save_audio", action="store_true", help="Don't save audio files")
    
    args = parser.parse_args()
    
    convert_shrutilipi(
        subset=args.subset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        save_audio=not args.no_save_audio,
        streaming=args.streaming
    )


if __name__ == "__main__":
    main()
