#!/usr/bin/env python3
"""
Convert STT dataset (audio→text) to TTS dataset (text→audio) format.
"""

import json
import os
from pathlib import Path
import soundfile as sf

def convert_stt_to_tts(stt_file, output_file, max_samples=None):
    """Convert STT format to TTS format with Mimi codec."""
    
    # Load STT data
    with open(stt_file, 'r', encoding='utf-8') as f:
        stt_data = json.load(f)
    
    print(f"Loaded {len(stt_data)} samples from {stt_file}")
    
    if max_samples:
        stt_data = stt_data[:max_samples]
        print(f"Using first {len(stt_data)} samples")
    
    # Initialize Mimi codec
    print("Loading Mimi codec...")
    try:
        from src.llamafactory.extras.mimi_codec import MimiCodec
        mimi = MimiCodec(model_id="kyutai/mimi")
    except ImportError as e:
        print(f"Error loading Mimi: {e}")
        print("Please install: pip install git+https://github.com/kyutai-labs/mimi.git")
        return
    
    # Convert to TTS format
    tts_data = []
    
    for i, item in enumerate(stt_data):
        try:
            # Extract text and audio path
            text = item["conversations"][1]["value"]  # GPT response
            audio_path = item["audios"][0]
            
            # Generate speaker ID from filename
            speaker_id = f"lahaja_{i % 10}"  # Cycle through 10 speakers
            
            # Encode audio with Mimi
            audio_codes = mimi.encode_file(audio_path)
            
            # Create TTS format entry
            tts_entry = {
                "conversations": [
                    {"from": "system", "value": "You are Qwen, a virtual human developed by Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."},
                    {"from": "human", "value": f"<|spk_start|>{speaker_id}<|spk_end|><|audio|>{text}"},
                    {"from": "gpt", "value": f"<|spk_start|>{speaker_id}<|spk_end|><|audio|>"}
                ],
                "audio_codes": audio_codes,
                "speaker": speaker_id
            }
            
            tts_data.append(tts_entry)
            
            if i % 50 == 0:
                print(f"Processed {i} samples...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save TTS data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tts_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(tts_data)} TTS samples to {output_file}")
    
    # Show sample
    if tts_data:
        sample = tts_data[0]
        print(f"\nSample TTS entry:")
        print(f"  Speaker: {sample['speaker']}")
        print(f"  Prompt: {sample['conversations'][1]['value'][:100]}...")
        print(f"  Response: {sample['conversations'][2]['value']}")
        print(f"  Audio codes: {len(sample['audio_codes'])} codebooks")
    
    return output_file

if __name__ == "__main__":
    # Convert your STT data to TTS format
    stt_file = "data/lahaja_test.json"
    tts_file = "data/lahaja_tts_speakers.json"
    
    convert_stt_to_tts(stt_file, tts_file, max_samples=1000)
    
    print(f"\n✅ TTS dataset ready: {tts_file}")
    print(f"Now you can train with: llamafactory-cli train examples/train_lora/qwen3omni_lora_tts_lahaja.yaml")
