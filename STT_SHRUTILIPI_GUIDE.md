# STT Training on Shrutilipi (Hindi) - Direct Streaming Guide

## 🎯 **What Was Modified**

### 1. **Audio Processing (mm_plugin.py)**
Modified `_regularize_audios()` to handle **audio bytes directly** from HuggingFace datasets:

```python
elif isinstance(audio, dict) and "bytes" in audio:
    # HuggingFace datasets format: {"bytes": bytes, "path": str}
    import io
    audio_buffer = io.BytesIO(audio["bytes"])
    audio_tensor, sr = torchaudio.load(audio_buffer)
    ...
```

### 2. **Dataset Configuration (dataset_info.json)**

```json
"shrutilipi_hindi": {
    "hf_hub_url": "ai4bharat/Shrutilipi",
    "subset": "hindi",
    "columns": {
        "prompt": "text",           # Target transcription
        "audio": "audio_filepath"   # Audio bytes (not path!)
    },
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt"
    }
}
```

**Important:** `audio_filepath` column contains **audio bytes** (dict with 'bytes' key), not file paths!

## 🚀 **Quick Start**

### 1. **Run Training**

```bash
llamafactory-cli train examples/train_lora/qwen25omni_stt_shrutilipi.yaml
```

### 2. **Training Config** (`qwen25omni_stt_shrutilipi.yaml`)

```yaml
model_name_or_path: Qwen/Qwen2.5-Omni-7B
template: qwen2_omni

# STT Mode - Thinker only
train_tts: false

# Direct streaming
dataset: shrutilipi_hindi
streaming: true
buffer_size: 65536

# Train audio encoder for Hindi
freeze_audio_encoder: false

# LoRA
finetuning_type: lora
lora_target: all
lora_rank: 32
lora_alpha: 64
```

## 📊 **Data Flow**

```
Shrutilipi Dataset (Streaming)
    ↓
audio_filepath: {"bytes": b'...', "path": "..."}
    ↓
mm_plugin._regularize_audios()
    ↓
Audio bytes → torchaudio.load(io.BytesIO(bytes))
    ↓
Audio encoder (Qwen2.5-Omni thinker)
    ↓
LLM → Transcription text
```

## 🔧 **How It Works**

### **LlamaFactory's Dataset Loading:**

1. **HF Dataset**: `load_dataset("ai4bharat/Shrutilipi", "hindi", streaming=True)`
2. **Column Mapping**:
   - `audio_filepath` → `audios` (contains bytes dict)
   - `text` → `prompt` (transcription target)
3. **Audio Processing**: 
   - Dict with `'bytes'` key detected by `mm_plugin`
   - `io.BytesIO()` creates buffer from bytes
   - `torchaudio.load()` reads the buffer
4. **Model Forward**: Audio → Thinker → Text output

## 📁 **Files Modified/Created**

| File | Change |
|------|--------|
| `src/llamafactory/data/mm_plugin.py` | Handle audio bytes (dict format) |
| `data/dataset_info.json` | Added `shrutilipi_hindi` entry |
| `examples/train_lora/qwen25omni_stt_shrutilipi.yaml` | Training config |

## ⚙️ **Key Configuration Details**

### **Streaming Settings**
- `streaming: true` - Enable HF dataset streaming
- `buffer_size: 65536` - Shuffle buffer (larger = better shuffle)
- `preprocessing_num_workers: null` - Disabled in streaming

### **Audio Encoder Training**
- `freeze_audio_encoder: false` - Train audio encoder for Hindi phonetics
- This is important because Qwen2.5-Omni was trained on English/Chinese primarily

### **LoRA Settings**
- `lora_rank: 32` - Higher rank for better adaptation
- `use_rslora: true` - Rank-stabilized LoRA for stability
- Target: `all` (includes audio projection layers if applicable)

## 💡 **Pro Tips**

### **1. Memory Management**
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
flash_attn: fa2
```

### **2. For Larger Datasets**
```yaml
buffer_size: 131072  # Even larger buffer
max_steps: 10000     # Limit steps instead of epochs
```

### **3. Multi-GPU Training**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train ...
```

### **4. Resuming Training**
```yaml
resume_from_checkpoint: saves/qwen25omni-lora-stt-hindi/checkpoint-500
```

## 🔍 **Verification**

### **Check Audio Bytes are Loading:**
```python
from datasets import load_dataset

ds = load_dataset("ai4bharat/Shrutilipi", "hindi", split="train", streaming=True)
sample = next(iter(ds))
print(type(sample["audio_filepath"]))  # Should be: <class 'dict'>
print(sample["audio_filepath"].keys())   # Should have: dict_keys(['path', 'bytes'])
print(type(sample["text"]))                # Should be: <class 'str'>
```

### **Check Training Logs:**
You should see:
- `audios` in batch (from mm_plugin)
- Audio being processed without file I/O errors
- Hindi text being generated as targets

## 🐛 **Troubleshooting**

| Issue | Solution |
|-------|----------|
| `RuntimeError: Invalid audio data` | Audio bytes not properly decoded - check `_regularize_audios` |
| `OOM during training` | Reduce `buffer_size`, enable `gradient_checkpointing` |
| `Slow data loading` | Increase `dataloader_num_workers` (if not streaming) |
| `Hindi text garbage` | Train longer, check if audio encoder is unfrozen |

## 🎓 **Architecture Summary**

**For STT (your case):**
- **Input**: Audio bytes from Shrutilipi
- **Model**: Thinker only (audio encoder + LLM)
- **Output**: Hindi transcription
- **Trainable**: Audio encoder (adapted to Hindi) + LLM LoRA

**Not used (TTS components):**
- Talker
- codec_head
- code2wav

## 📊 **Expected Results**

After training on Shrutilipi (~50GB Hindi audio):
- Model learns Hindi phonetics
- Better ASR performance on Hindi speech
- Can be combined with other Hindi text corpora

**Good luck with Hindi STT training! 🇮🇳**
