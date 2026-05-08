# TTS Training for Qwen2.5-Omni in LlamaFactory

## Overview

This implementation enables Text-to-Speech (TTS) training for Qwen2.5-Omni within LlamaFactory. The goal is to train the **Thinker** component to output audio codec tokens that the **Talker.codec_head** can decode into speech.

## Key Architecture Changes

### 1. Model Loading (`src/llamafactory/model/loader.py`)

**Standard Mode** (audio understanding):
- Only loads `model.thinker` (audio encoder + LLM backbone)
- Talker is completely dropped

**TTS Mode** (`train_tts: true`):
- Loads **FULL** model (thinker + talker)
- **Frozen**: `audio_tower` (audio encoder) - optional
- **Frozen**: `talker.code2wav` (audio decoder) - always frozen
- **Trainable**: `talker.codec_head` - this is what we train!

### 2. TTS Arguments (`src/llamafactory/hparams/finetuning_args.py`)

```yaml
train_tts: true                    # Enable TTS training mode
freeze_audio_encoder: true           # Freeze audio encoder (default)
freeze_codec_head: false           # Train codec head (default)
audio_codec_vocab_size: 16000      # Codec vocabulary size
```

### 3. Dataset Processing

#### TTSDatasetProcessor (`src/llamafactory/data/processor/tts_processor.py`)
- Handles `audio_codes` column as training targets
- Masks text labels with `IGNORE_INDEX`
- Passes codec tokens through as `audio_code_targets`

#### Data Collator (`src/llamafactory/data/collator.py`)
- Batches `audio_code_targets` from dataset
- Pads codec token sequences to batch max length

### 4. TTS Trainer (`src/llamafactory/train/sft/tts_trainer.py`)

- Computes loss on `talker.codec_head` outputs
- Compares predicted codec tokens vs Mimi targets
- Loss computed per codebook with proper masking

### 5. Mimi Codec (`src/llamafactory/extras/mimi_codec.py`)

**Purpose**: Extract target audio codes from ground truth audio.
**Note**: Mimi is used ONLY for dataset preparation, NOT for training!

```python
mimi = MimiCodec(model_id="kyutai/mimi")
audio_codes = mimi.encode_file("audio.wav")  # [32 codebooks, num_frames]
```

## Dataset Format

### Input Format (CSV/JSON):
```csv
text,audio_path
"Hello world","audio1.wav"
"How are you?","audio2.wav"
```

### Output Format (LlamaFactory ShareGPT):
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Generate speech: Hello world"},
      {"from": "gpt", "value": "<|audio|>"}
    ],
    "audio_codes": [[123, 456, ...], [234, 567, ...], ...]  // 32 codebooks
  }
]
```

## Usage

### 1. Prepare Dataset with Mimi Codec

```bash
python scripts/convert_tts_dataset.py \
    --dataset_path /path/to/audio/files \
    --metadata_file metadata.csv \
    --output_file tts_dataset.json \
    --text_column text \
    --audio_column audio_path \
    --mimi_model kyutai/mimi
```

### 2. Add to dataset_info.json

```json
{
  "tts_dataset": {
    "file_name": "tts_dataset.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "audio_codes": "audio_codes"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
```

### 3. Train with TTS Config

```bash
llamafactory-cli train examples/train_lora/qwen25omni_lora_tts.yaml
```

## Training Config Example

```yaml
model_name_or_path: Qwen/Qwen2.5-Omni-7B
template: qwen2_omni

# TTS Mode
train_tts: true
freeze_audio_encoder: true
freeze_codec_head: false

# LoRA for Thinker LLM backbone
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32

# Dataset
dataset: tts_dataset

# Training
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 3.0
```

## Files Modified/Created

### Core Implementation:
1. `src/llamafactory/hparams/finetuning_args.py` - TTS arguments
2. `src/llamafactory/model/loader.py` - Keep talker for TTS
3. `src/llamafactory/data/processor/tts_processor.py` - TTS dataset processor
4. `src/llamafactory/data/processor/__init__.py` - Export TTSDatasetProcessor
5. `src/llamafactory/data/loader.py` - Wire up TTS processor
6. `src/llamafactory/data/collator.py` - Batch audio codec targets
7. `src/llamafactory/train/sft/tts_trainer.py` - TTS loss computation
8. `src/llamafactory/train/sft/workflow.py` - Use TTSTrainer
9. `src/llamafactory/extras/mimi_codec.py` - Mimi integration
10. `src/llamafactory/extras/packages.py` - is_mimi_available()

### Scripts & Config:
1. `scripts/convert_tts_dataset.py` - Convert audio to codec tokens
2. `examples/train_lora/qwen25omni_lora_tts.yaml` - TTS training config

## What Gets Trained?

| Component | Standard SFT | TTS Training |
|-----------|--------------|--------------|
| Thinker LLM | ✅ Trainable | ✅ Trainable (LoRA) |
| Audio Encoder | ✅ Trainable | ❌ Frozen |
| Talker.codec_head | ❌ Dropped | ✅ **Trainable** |
| Talker.code2wav | ❌ Dropped | ❌ **Frozen** |

## Important Notes

1. **Mimi codec is read-only**: We use it only to extract target codes from audio files
2. **Audio encoder frozen**: We don't train audio understanding, only audio generation
3. **code2wav frozen**: We don't train the audio decoder, only the codec head
4. **Thinker + codec_head trainable**: These are the only components that learn

## Testing Checklist

- [ ] Mimi codec can encode audio files
- [ ] Dataset converter produces correct JSON format
- [ ] Model loads full Qwen2.5-Omni (not just thinker) in TTS mode
- [ ] TTSDatasetProcessor handles audio_codes column
- [ ] Collator batches audio_code_targets correctly
- [ ] TTSTrainer computes loss on codec_head outputs
- [ ] Training runs without errors
- [ ] Loss decreases over training steps
