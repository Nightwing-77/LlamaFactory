
# Convert dataset
import json, os
from pathlib import Path
from datasets import load_dataset
import soundfile as sf

def convert_lahaja(split="test", max_samples=None):
    dataset = load_dataset("ai4bharat/Lahaja", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    Path("./lahaja_audio").mkdir(exist_ok=True)
    converted = []
    
    for i, ex in enumerate(dataset):
        text = ex.get("text", "").strip()
        if not text: continue
        
        # Save audio
        audio_array = ex.get("audio_filepath", {}).get("array")
        sr = ex.get("audio_filepath", {}).get("sampling_rate", 16000)
        if audio_array is not None:
            audio_path = f"./lahaja_audio/lahaja_{i:08d}.wav"
            sf.write(audio_path, audio_array, sr)
        
            converted.append({
                "conversations": [
                    {"from": "human", "value": "<audio>\nTranscribe this audio"},
                    {"from": "gpt", "value": text}
                ],
                "audios": [audio_path]
            })
    
    os.makedirs("data", exist_ok=True)
    with open(f"data/lahaja_{split}.json", "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(converted)} examples")
    return converted

# Run
train_data = convert_lahaja("test")
