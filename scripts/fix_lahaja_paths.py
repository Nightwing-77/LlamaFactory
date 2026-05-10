#!/usr/bin/env python3
"""Fix audio paths in lahaja_test.json to use absolute paths."""

import json
import os

# Get absolute path to audio folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audio_dir = os.path.join(base_dir, "lahaja_audio")

print(f"Audio directory: {audio_dir}")

# Load JSON
json_path = os.path.join(base_dir, "data", "lahaja_test.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Replace paths
for item in data:
    item['audios'] = [os.path.join(audio_dir, os.path.basename(a)) for a in item['audios']]

# Save
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated {len(data)} entries with absolute paths")
print(f"\nSample: {data[0]['audios']}")
