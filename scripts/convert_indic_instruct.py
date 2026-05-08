#!/usr/bin/env python3
"""
Convert ai4bharat/indic-instruct-data-v0.1 (anudesh subset) to LlamaFactory training format.
Uses 'hi' split + 20% of 'en' split for Qwen2.5-Omni fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    raise


def convert_messages_to_sharegpt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert HuggingFace messages format to ShareGPT conversations format."""
    conversations = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # Map roles: user -> human, assistant -> gpt
        if role == "user":
            conversations.append({"from": "human", "value": content})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": content})
        elif role == "system":
            conversations.append({"from": "system", "value": content})
        else:
            conversations.append({"from": role, "value": content})
    
    return conversations


def process_dataset(output_path: str = "indic_instruct_qwen25_omni.json", seed: int = 42):
    """
    Load anudesh dataset, combine hi + 20% en, and convert to LlamaFactory format.
    
    Args:
        output_path: Path to save the output JSON file
        seed: Random seed for sampling 20% of en split
    """
    random.seed(seed)
    
    print("Loading ai4bharat/indic-instruct-data-v0.1 dataset...")
    
    # Load Hindi split (hi) - complete dataset
    print("Loading 'hi' (Hindi) split...")
    hi_dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", name="anudesh", split="hi")
    print(f"  Loaded {len(hi_dataset)} Hindi examples")
    
    # Load English split (en) - sample 20%
    print("Loading 'en' (English) split...")
    en_dataset = load_dataset("ai4bharat/indic-instruct-data-v0.1", name="anudesh", split="en")
    print(f"  Loaded {len(en_dataset)} English examples")
    
    # Sample 20% of English data
    en_sample_size = int(len(en_dataset) * 0.20)
    en_indices = random.sample(range(len(en_dataset)), en_sample_size)
    en_sampled = en_dataset.select(en_indices)
    print(f"  Sampled {en_sample_size} English examples (20%)")
    
    # Combine datasets
    combined = []
    
    # Process Hindi data
    print("\nProcessing Hindi data...")
    for i, example in enumerate(hi_dataset):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(hi_dataset)} Hindi examples...")
        
        conversations = convert_messages_to_sharegpt(example["messages"])
        
        # Only include examples with valid conversation structure
        if len(conversations) >= 2 and any(c["from"] == "human" for c in conversations):
            combined.append({
                "id": example.get("id", f"hi_{i}"),
                "conversations": conversations,
                "language": "hi",
                "num_turns": example.get("num_turns", len([c for c in conversations if c["from"] == "human"]))
            })
    
    # Process English data
    print(f"\nProcessing English data...")
    for i, example in enumerate(en_sampled):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(en_sampled)} English examples...")
        
        conversations = convert_messages_to_sharegpt(example["messages"])
        
        # Only include examples with valid conversation structure
        if len(conversations) >= 2 and any(c["from"] == "human" for c in conversations):
            combined.append({
                "id": example.get("id", f"en_{i}"),
                "conversations": conversations,
                "language": "en",
                "num_turns": example.get("num_turns", len([c for c in conversations if c["from"] == "human"]))
            })
    
    print(f"\nTotal valid examples: {len(combined)}")
    print(f"  Hindi: {len([x for x in combined if x['language'] == 'hi'])}")
    print(f"  English: {len([x for x in combined if x['language'] == 'en'])}")
    
    # Shuffle the combined dataset
    random.shuffle(combined)
    
    # Save to JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset saved to: {output_file.absolute()}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print sample example
    print("\n" + "="*60)
    print("Sample example:")
    print("="*60)
    sample = combined[0]
    print(f"ID: {sample['id']}")
    print(f"Language: {sample['language']}")
    print(f"Turns: {sample['num_turns']}")
    print("Conversations:")
    for conv in sample['conversations'][:4]:  # Show first 4 turns
        content = conv['value'][:100] + "..." if len(conv['value']) > 100 else conv['value']
        print(f"  {conv['from']}: {content}")
    
    return combined


def create_dataset_info_entry(output_json_path: str = "indic_instruct_qwen25_omni.json"):
    """
    Print the dataset_info.json entry needed for LlamaFactory.
    """
    print("\n" + "="*60)
    print("Add this to your dataset_info.json:")
    print("="*60)
    
    entry = {
        "indic_instruct_qwen25_omni": {
            "file_name": output_json_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "language": ["hi", "en"],
                "task": "sft",
                "source": "ai4bharat/indic-instruct-data-v0.1"
            }
        }
    }
    
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    
    return entry


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ai4bharat dataset to LlamaFactory format")
    parser.add_argument(
        "--output", 
        type=str, 
        default="indic_instruct_qwen25_omni.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for sampling English data"
    )
    parser.add_argument(
        "--dataset-info-only",
        action="store_true",
        help="Only print dataset_info.json entry"
    )
    
    args = parser.parse_args()
    
    if not args.dataset_info_only:
        # Process the dataset
        process_dataset(output_path=args.output, seed=args.seed)
    
    # Print dataset_info entry
    create_dataset_info_entry(output_json_path=args.output)
