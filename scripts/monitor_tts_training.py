#!/usr/bin/env python3
"""Monitor TTS training progress and validate voice quality checkpoints.

This script helps you track:
1. Training loss curve
2. Validation loss
3. Gradient norms
4. Learning rate schedule
5. Speaker-specific metrics

Usage:
    # During training - live monitoring
    python scripts/monitor_tts_training.py --checkpoint_dir saves/qwen25omni-7b/lora/tts-hindi-speakers --watch
    
    # After training - analyze results
    python scripts/monitor_tts_training.py --checkpoint_dir saves/qwen25omni-7b/lora/tts-hindi-speakers --analyze
    
    # Test voice generation at checkpoint
    python scripts/monitor_tts_training.py --checkpoint_dir saves/qwen25omni-7b/lora/tts-hindi-speakers/checkpoint-1000 --test_voice --speaker kavya --text "नमस्ते, मैं काव्या हूँ"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from transformers import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor TTS training")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--watch", action="store_true", help="Watch for updates during training")
    parser.add_argument("--analyze", action="store_true", help="Analyze training results")
    parser.add_argument("--test_voice", action="store_true", help="Test voice generation")
    parser.add_argument("--speaker", type=str, default="kavya", help="Speaker to test")
    parser.add_argument("--text", type=str, default="नमस्ते, यह एक परीक्षण है।", help="Text to synthesize")
    parser.add_argument("--device", type=str, default="cuda", help="Device for testing")
    return parser.parse_args()


def parse_log_file(log_file: str):
    """Parse training log file for metrics."""
    metrics = {
        'steps': [],
        'loss': [],
        'learning_rate': [],
        'grad_norm': [],
        'eval_loss': [],
        'eval_steps': []
    }
    
    if not os.path.exists(log_file):
        return metrics
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                # Parse JSON log lines
                if '{' in line:
                    data = json.loads(line[line.index('{'):])
                    
                    if 'loss' in data:
                        metrics['steps'].append(data.get('step', 0))
                        metrics['loss'].append(data['loss'])
                        metrics['learning_rate'].append(data.get('learning_rate', 0))
                        metrics['grad_norm'].append(data.get('grad_norm', 0))
                    
                    if 'eval_loss' in data:
                        metrics['eval_steps'].append(data.get('step', 0))
                        metrics['eval_loss'].append(data['eval_loss'])
            except:
                continue
    
    return metrics


def plot_training_curves(metrics: dict, output_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curve
    if metrics['steps'] and metrics['loss']:
        axes[0, 0].plot(metrics['steps'], metrics['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # Learning rate
    if metrics['steps'] and metrics['learning_rate']:
        axes[0, 1].plot(metrics['steps'], metrics['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].grid(True)
    
    # Gradient norm
    if metrics['steps'] and metrics['grad_norm']:
        axes[1, 0].plot(metrics['steps'], metrics['grad_norm'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Grad Norm')
        axes[1, 0].grid(True)
    
    # Validation loss
    if metrics['eval_steps'] and metrics['eval_loss']:
        axes[1, 1].plot(metrics['eval_steps'], metrics['eval_loss'], 'ro-')
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Eval Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved training curves to {output_path}")


def check_training_health(metrics: dict):
    """Check if training is healthy."""
    health_report = []
    
    if not metrics['loss']:
        health_report.append("⚠️ No training loss found - training may not have started")
        return health_report
    
    # Check loss trend
    recent_losses = metrics['loss'][-100:] if len(metrics['loss']) > 100 else metrics['loss']
    if len(recent_losses) > 10:
        early_avg = np.mean(recent_losses[:10])
        late_avg = np.mean(recent_losses[-10:])
        
        if late_avg < early_avg * 0.9:
            health_report.append(f"✅ Loss is decreasing: {early_avg:.4f} → {late_avg:.4f}")
        elif late_avg > early_avg * 1.1:
            health_report.append(f"⚠️ Loss is increasing: {early_avg:.4f} → {late_avg:.4f} (overfitting?)")
        else:
            health_report.append(f"ℹ️ Loss is stable: {early_avg:.4f} → {late_avg:.4f}")
    
    # Check gradient norms
    if metrics['grad_norm']:
        recent_grads = metrics['grad_norm'][-50:]
        if np.mean(recent_grads) > 10:
            health_report.append(f"⚠️ High gradient norms detected (avg: {np.mean(recent_grads):.2f}) - may be unstable")
        elif np.mean(recent_grads) < 0.001:
            health_report.append(f"⚠️ Very low gradient norms (avg: {np.mean(recent_grads):.4f}) - may be stuck")
        else:
            health_report.append(f"✅ Gradient norms look healthy (avg: {np.mean(recent_grads):.4f})")
    
    # Check eval loss
    if metrics['eval_loss'] and len(metrics['eval_loss']) > 1:
        if metrics['eval_loss'][-1] > metrics['eval_loss'][0] * 1.1:
            health_report.append(f"⚠️ Validation loss increasing - possible overfitting")
        else:
            health_report.append(f"✅ Validation loss is stable or improving")
    
    return health_report


def list_checkpoints(checkpoint_dir: str):
    """List available checkpoints."""
    checkpoints = []
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    for item in os.listdir(checkpoint_dir):
        if item.startswith('checkpoint-'):
            step = int(item.split('-')[1])
            checkpoints.append((step, os.path.join(checkpoint_dir, item)))
    
    return sorted(checkpoints)


def test_voice_generation(checkpoint_path: str, speaker: str, text: str, device: str = "cuda"):
    """Test voice generation at a checkpoint."""
    try:
        from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
        
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        model = Qwen2_5OmniModel.from_pretrained(
            checkpoint_path,
            device_map=device,
            torch_dtype="auto",
            attn_implementation="flash_attention_2"
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(checkpoint_path)
        
        # Prepare input with speaker token
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate speech in {speaker}'s voice: {text}"}
        ]
        
        text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text_input], return_tensors="pt")
        inputs = inputs.to(model.device)
        
        print(f"Generating speech for: \"{text}\" in {speaker}'s voice...")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024)
        
        # Extract audio tokens (if available in output)
        if hasattr(outputs, 'audio_tokens'):
            print(f"✅ Generated audio tokens: {outputs.audio_tokens.shape}")
            print("Voice generation successful!")
        else:
            print("ℹ️ Generation completed - check output for audio")
        
    except Exception as e:
        print(f"❌ Error testing voice: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    
    # Find log file
    log_file = os.path.join(args.checkpoint_dir, "training.log")
    if not os.path.exists(log_file):
        # Try to find any log file
        for root, dirs, files in os.walk(args.checkpoint_dir):
            for f in files:
                if f.endswith('.log'):
                    log_file = os.path.join(root, f)
                    break
    
    if args.watch:
        print(f"👀 Watching {args.checkpoint_dir} for training updates...")
        print("Press Ctrl+C to stop\n")
        
        last_mtime = 0
        try:
            while True:
                if os.path.exists(log_file):
                    mtime = os.path.getmtime(log_file)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        metrics = parse_log_file(log_file)
                        
                        print(f"\n📊 Training Status at step {metrics['steps'][-1] if metrics['steps'] else 0}:")
                        health = check_training_health(metrics)
                        for item in health:
                            print(f"  {item}")
                        
                        # List checkpoints
                        checkpoints = list_checkpoints(args.checkpoint_dir)
                        if checkpoints:
                            print(f"\n💾 Checkpoints: {len(checkpoints)} saved (latest: step {checkpoints[-1][0]})")
                        
                        # Plot curves
                        if metrics['steps']:
                            plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
                            plot_training_curves(metrics, plot_path)
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n\nStopped watching.")
    
    elif args.analyze:
        print(f"📈 Analyzing training results from {args.checkpoint_dir}\n")
        
        metrics = parse_log_file(log_file)
        
        if not metrics['loss']:
            print("❌ No training data found")
            return
        
        print(f"Total steps: {len(metrics['steps'])}")
        print(f"Initial loss: {metrics['loss'][0]:.4f}")
        print(f"Final loss: {metrics['loss'][-1]:.4f}")
        print(f"Best loss: {min(metrics['loss']):.4f}")
        
        if metrics['eval_loss']:
            print(f"\nValidation metrics:")
            print(f"  Initial: {metrics['eval_loss'][0]:.4f}")
            print(f"  Final: {metrics['eval_loss'][-1]:.4f}")
            print(f"  Best: {min(metrics['eval_loss']):.4f}")
        
        print("\nHealth Report:")
        health = check_training_health(metrics)
        for item in health:
            print(f"  {item}")
        
        # Plot final curves
        plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
        plot_training_curves(metrics, plot_path)
        
        # List all checkpoints
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if checkpoints:
            print(f"\n💾 Available Checkpoints:")
            for step, path in checkpoints:
                print(f"  - Step {step}: {path}")
    
    elif args.test_voice:
        test_voice_generation(args.checkpoint_dir, args.speaker, args.text, args.device)
    
    else:
        print("Please specify --watch, --analyze, or --test_voice")
        print(f"\nAvailable checkpoints in {args.checkpoint_dir}:")
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if checkpoints:
            for step, path in checkpoints:
                print(f"  - Step {step}")
        else:
            print("  No checkpoints found yet")


if __name__ == "__main__":
    main()
