# TTS Voice/Speaker Training Guide

## 🎙️ **Voice Support in Qwen2.5-Omni**

Qwen2.5-Omni supports **built-in speaker tokens** for voice conditioning:
- `<|spk_start|>{speaker_id}<|spk_end|>` - Controls voice characteristics
- Default voices: `Chelsie`, `Serena` (and others)

## 📊 **Dataset Format with Speaker IDs**

### **Input Format (CSV)**
```csv
text,audio_path,speaker_id
"Hello world","audio1.wav","Chelsie"
"Namaste","audio2.wav","Serena"
"How are you?","audio3.wav","Chelsie"
```

### **Output Format (JSON with Speaker Tokens)**
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Generate speech: Hello world"},
      {"from": "gpt", "value": "<|spk_start|>Chelsie<|spk_end|><|audio|>"}
    ],
    "audio_codes": [[123, 456, ...], [234, 567, ...], ...],
    "speaker": "Chelsie"
  },
  {
    "conversations": [
      {"from": "human", "value": "Generate speech: Namaste"},
      {"from": "gpt", "value": "<|spk_start|>Serena<|spk_end|><|audio|>"}
    ],
    "audio_codes": [[789, 12, ...], [345, 678, ...], ...],
    "speaker": "Serena"
  }
]
```

## 🚀 **Usage**

### **1. Convert Dataset with Speaker IDs**

```bash
python scripts/convert_tts_dataset.py \
    --dataset_path /path/to/audio/files \
    --metadata_file metadata.csv \
    --output_file tts_dataset.json \
    --text_column text \
    --audio_column audio_path \
    --speaker_column speaker_id \
    --use_speaker_tokens \
    --mimi_model kyutai/mimi
```

### **2. Add to dataset_info.json**

```json
{
  "tts_voice_dataset": {
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

### **3. Train with Voice Support**

```yaml
model_name_or_path: Qwen/Qwen2.5-Omni-7B
template: qwen2_omni  # Has spk_start/spk_end tokens

# TTS Mode
train_tts: true
freeze_audio_encoder: true
freeze_codec_head: false

# Dataset with speaker info
dataset: tts_voice_dataset

# LoRA settings
finetuning_type: lora
lora_target: all
lora_rank: 16

# Training
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 3.0
```

## 🔊 **How Speaker Tokens Work**

### **Architecture Flow:**

```
Input Text + Speaker Token
         ↓
    [Thinker LLM]
         ↓
    [Hidden States] + Speaker Embedding
         ↓
    [Talker.codec_head]
         ↓
    [Audio Codec Tokens] → [code2wav] → Audio
```

### **What Gets Trained:**

| Component | Voice Training |
|-----------|----------------|
| Thinker LLM | ✅ Trained (LoRA) |
| Speaker Embedding | ✅ Implicitly learned via tokens |
| Codec Head | ✅ Trained |
| Audio Encoder | ❌ Frozen |
| code2wav | ❌ Frozen |

## 🎯 **Custom Voices**

You can train **custom speaker IDs** beyond Chelsie/Serena:

### **Example: Train "MyVoice"**

```csv
text,audio_path,speaker_id
"Hello","myvoice_1.wav","MyVoice"
"How are you?","myvoice_2.wav","MyVoice"
"Good morning","myvoice_3.wav","MyVoice"
```

The model will learn to associate `MyVoice` token with your voice characteristics!

## 📁 **Key Files for Voice Training**

1. **`scripts/convert_tts_dataset.py`** - Now supports `--speaker_column` and `--use_speaker_tokens`
2. **`src/llamafactory/data/template.py`** - `qwen2_omni` template has `spk_start_token` and `spk_end_token`
3. **`src/llamafactory/data/mm_plugin.py`** - `Qwen2OmniPlugin` handles speaker tokens

## ✅ **Voice Training Checklist**

- [ ] Audio files organized by speaker
- [ ] CSV with `speaker_id` column
- [ ] Run converter with `--use_speaker_tokens`
- [ ] Verify speaker tokens in output: `<|spk_start|>SpeakerName<|spk_end|>`
- [ ] Train with `train_tts: true`
- [ ] During inference, specify speaker: `Generate speech in Chelsie's voice: Hello`

## 📝 **Inference Example**

After training, generate speech with specific voices:

```python
# Generate with trained voice
conversation = [
    {"role": "user", "content": "Say 'Hello' in Chelsie's voice"}
]

# Or use speaker tokens directly
conversation = [
    {"role": "user", "content": "<|spk_start|>Chelsie<|spk_end|>Hello"}
]
```

**Note:** The speaker conditioning happens through the text prompt. The model learns to generate different voice characteristics based on the speaker token provided!
