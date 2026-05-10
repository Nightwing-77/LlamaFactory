# Hindi TTS Speaker Training Guide - Voice Cloning

Complete guide for training custom Hindi voices (kavya, agastya, etc.) using LlamaFactory with Qwen2.5-Omni.

## 📋 Overview

This guide covers:
1. Converting your pipe-separated CSV dataset to TTS format
2. Setting up LoRA training for thinker-only TTS
3. Adding new speakers (kavya, agastya)
4. Monitoring training progress
5. Voice cloning best practices

## 🎯 Architecture

```
Input Text + Speaker Token
         ↓
    [Thinker LLM] ← LoRA Training Here
         ↓
    [Hidden States] + Speaker Embedding
         ↓
    [Talker.codec_head] ← Also Trained
         ↓
    [Audio Codec Tokens] → [code2wav] → Speech
```

**Frozen Components:**
- `audio_tower` (audio encoder) - not needed for TTS generation
- `code2wav` (audio decoder) - uses pretrained Mimi codec

**Trainable Components:**
- `thinker` (LLM backbone) - via LoRA adapters
- `talker.codec_head` - maps to codec vocabulary

## 🚀 Step-by-Step Training

### Step 1: Convert Your Dataset

Your CSV format:
```
kavya/00002.wav|kavya|बता दें कि ये फोटो ड्रग्स पैडलर द्वारा...
agastya/00002.wav|agastya|बता दें कि ये फोटो ड्रग्स पैडलर द्वारा...
```

Convert to TTS format:
```bash
python scripts/convert_hindi_tts_dataset.py \
    --csv_file your_dataset.csv \
    --audio_base_path /path/to/audio/files \
    --output_file data/hindi_tts_speakers.json \
    --mimi_model kyutai/mimi
```

**Output format (data/hindi_tts_speakers.json):**
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Generate speech in kavya's voice: बता दें कि ये फोटो..."},
      {"from": "gpt", "value": "<|spk_start|>kavya<|spk_end|><|audio|>"}
    ],
    "audio_codes": [[123, 456, ...], [234, 567, ...], ...],
    "speaker": "kavya"
  }
]
```

### Step 2: Verify Dataset Registration

Your dataset is already registered in `data/dataset_info.json`:
```json
"hindi_tts_speakers": {
  "file_name": "hindi_tts_speakers.json",
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
```

### Step 3: Start Training

```bash
llamafactory-cli train examples/train_lora/qwen25omni_lora_tts_hindi.yaml
```

## 📊 Monitoring Training

### Live Monitoring (Recommended)
```bash
# Terminal 1: Start training
llamafactory-cli train examples/train_lora/qwen25omni_lora_tts_hindi.yaml

# Terminal 2: Monitor progress
python scripts/monitor_tts_training.py \
    --checkpoint_dir saves/qwen25omni-7b/lora/tts-hindi-speakers \
    --watch
```

### Check Training Health

The monitor will show:
- ✅ Loss decreasing: 2.3412 → 1.8923
- ✅ Gradient norms healthy (avg: 0.4521)
- ✅ Validation loss stable
- ⚠️ Loss increasing - possible overfitting

### Expected Metrics

| Stage | Loss Range | Notes |
|-------|------------|-------|
| Initial | 3.0-4.0 | Random initialization |
| After 500 steps | 1.5-2.5 | Voice emerging |
| After 2000 steps | 0.8-1.5 | Good quality |
| Converged | 0.3-0.8 | Excellent quality |

### WandB Integration

Set your API key:
```bash
export WANDB_API_KEY=your_key_here
```

Or add to config:
```yaml
report_to: wandb
wandb_project: qwen25-omni-tts-hindi
wandb_run_name: kavya-agastya-voices
```

## 🎙️ Adding New Speakers

### Speaker Token Format

Qwen2.5-Omni uses special tokens for speaker conditioning:
- `<|spk_start|>{speaker_name}<|spk_end|><|audio|>`

**Built-in speakers:** Chelsie, Serena (pre-trained)

**Custom speakers:** kavya, agastya, any_name_you_want

### Training Strategy for Multiple Speakers

1. **Balanced Dataset**: Ensure each speaker has ~equal samples
   ```
   kavya: 5000 samples
   agastya: 5000 samples
   ```

2. **Speaker Distribution in Prompts**:
   - System learns to associate speaker token with voice characteristics
   - Prompt format: `Generate speech in {speaker}'s voice: {text}`

3. **Training Convergence**:
   - Single speaker: ~1000-2000 steps
   - Two speakers: ~3000-5000 steps
   - Three+ speakers: ~5000-10000 steps

### Testing Different Voices

After training, test voice generation:

```python
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor

model = Qwen2_5OmniModel.from_pretrained("saves/qwen25omni-7b/lora/tts-hindi-speakers/checkpoint-3000")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# Test kavya's voice
conversation = [
    {"role": "user", "content": "Generate speech in kavya's voice: नमस्ते, मैं काव्या हूँ"}
]

# Test agastya's voice
conversation = [
    {"role": "user", "content": "Generate speech in agastya's voice: नमस्ते, मैं अगस्त्य हूँ"}
]
```

## 🔧 Hyperparameters Explained

### LoRA Settings (`examples/train_lora/qwen25omni_lora_tts_hindi.yaml`)

```yaml
lora_rank: 32          # Rank for LoRA matrices (higher = more capacity)
lora_alpha: 64         # Scaling factor (typically 2x rank)
lora_dropout: 0.05     # Regularization
use_rslora: true       # Rank-stabilized LoRA (better for large ranks)
```

**Why these values?**
- `rank=32`: Higher capacity needed for voice characteristics
- `alpha=64`: Standard 2x scaling for stable training
- `rslora=true`: Essential for rank > 16 to prevent instability

### Training Settings

```yaml
learning_rate: 5.0e-5       # Higher than standard SFT (needs more adaptation)
num_train_epochs: 5.0       # More epochs for voice learning
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Effective batch size = 8
warmup_ratio: 0.1           # 10% warmup for stable start
```

**Why 5 epochs?**
- Voice cloning requires more iterations than text tasks
- Each epoch sees all speaker samples
- Convergence typically at 3-5 epochs

### TTS-Specific Settings

```yaml
train_tts: true             # Enable TTS mode (keeps talker.codec_head)
freeze_audio_encoder: true  # Don't train audio understanding
freeze_codec_head: false    # TRAIN the codec head
audio_codec_vocab_size: 16000
```

## 📈 Troubleshooting

### Problem: Loss not decreasing
**Solutions:**
1. Check `audio_codes` are valid Mimi tokens
2. Verify speaker tokens in dataset: `<|spk_start|>kavya<|spk_end|>`
3. Increase learning rate: `learning_rate: 1.0e-4`
4. Check gradient norms - if zero, model not training

### Problem: All voices sound the same
**Solutions:**
1. Ensure speaker column is properly included in dataset
2. Add speaker name to prompt: `"Generate speech in {speaker}'s voice: {text}"`
3. Train longer (more epochs)
4. Increase LoRA rank: `lora_rank: 64`

### Problem: Poor audio quality
**Solutions:**
1. Verify Mimi codec tokens are correctly extracted
2. Check audio sample rate (should be 24kHz)
3. Train more steps (minimum 2000 for decent quality)
4. Increase `cutoff_len` if text is long

### Problem: OOM (Out of Memory)
**Solutions:**
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Increase to maintain effective batch size
gradient_checkpointing: true
flash_attn: disabled  # If fa2 causes issues
```

## 🎓 Advanced: Fine-Tuning Strategy

### Phase 1: Single Speaker Warmup (Optional)
1. Train on just "kavya" for 1000 steps
2. This establishes base voice quality
3. Then add "agastya" for multi-speaker training

### Phase 2: Multi-Speaker Training
1. Combine all speakers
2. Train with balanced sampling
3. Monitor speaker-specific loss if possible

### Phase 3: Fine-Tuning
1. Lower learning rate: `2.0e-5`
2. More epochs for refinement
3. Evaluate on held-out sentences

## 📝 Inference After Training

### Export LoRA Adapter
```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --adapter_name_or_path saves/qwen25omni-7b/lora/tts-hindi-speakers \
    --export_dir exports/qwen25omni-tts-hindi \
    --template qwen2_omni \
    --finetuning_type lora
```

### Generate Speech
```python
import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor

# Load base model + LoRA
model = Qwen2_5OmniModel.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load your trained adapter
model.load_adapter("exports/qwen25omni-tts-hindi")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# Generate in kavya's voice
conversation = [
    {"role": "user", "content": "Generate speech in kavya's voice: नमस्ते, आप कैसे हैं?"}
]
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt")
inputs = inputs.to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
audio = outputs.audio  # Generated audio waveform
```

## ✅ Checklist

Before training:
- [ ] CSV file with path|speaker|text format
- [ ] Audio files in correct location
- [ ] Mimi codec installed: `pip install git+https://github.com/kyutai-labs/mimi.git`
- [ ] Dataset converted to JSON format
- [ ] `data/hindi_tts_speakers.json` exists
- [ ] `data/dataset_info.json` updated
- [ ] Config file reviewed
- [ ] WandB API key set (optional but recommended)

During training:
- [ ] Loss is decreasing
- [ ] Gradient norms are stable (0.1-10 range)
- [ ] Validation loss not increasing
- [ ] Checkpoints are saving

After training:
- [ ] Export LoRA adapter
- [ ] Test voice generation for each speaker
- [ ] Compare audio quality
- [ ] Save best checkpoint

## 🔗 Quick Commands

```bash
# 1. Convert dataset
python scripts/convert_hindi_tts_dataset.py \
    --csv_file dataset.csv \
    --audio_base_path /path/to/audio \
    --output_file data/hindi_tts_speakers.json

# 2. Train
llamafactory-cli train examples/train_lora/qwen25omni_lora_tts_hindi.yaml

# 3. Monitor
python scripts/monitor_tts_training.py \
    --checkpoint_dir saves/qwen25omni-7b/lora/tts-hindi-speakers \
    --watch

# 4. Test voice
python scripts/monitor_tts_training.py \
    --checkpoint_dir saves/qwen25omni-7b/lora/tts-hindi-speakers/checkpoint-3000 \
    --test_voice \
    --speaker kavya \
    --text "नमस्ते, यह एक परीक्षण है।"

# 5. Export
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --adapter_name_or_path saves/qwen25omni-7b/lora/tts-hindi-speakers \
    --export_dir exports/qwen25omni-tts-hindi \
    --template qwen2_omni \
    --finetuning_type lora
```

---

**Questions?** Check `TTS_TRAINING.md` and `TTS_VOICE_TRAINING.md` for more details.
