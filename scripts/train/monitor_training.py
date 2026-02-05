#!/usr/bin/env python3
"""
Monitor training progress and automatically adjust if too slow.
"""

import time
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_training_progress():
    """Check current training progress."""
    history_file = Path("results/training/training_history.json")
    
    if not history_file.exists():
        return None, "Training not started yet"
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        epochs_completed = len(data.get("train_losses", []))
        latest_train_loss = data.get("train_losses", [])[-1] if data.get("train_losses") else None
        latest_val_loss = data.get("val_losses", [])[-1] if data.get("val_losses") else None
        
        return {
            "epochs_completed": epochs_completed,
            "train_loss": latest_train_loss,
            "val_loss": latest_val_loss
        }, "OK"
    except Exception as e:
        return None, str(e)

def estimate_time_per_epoch(start_time, epochs_completed):
    """Estimate time per epoch based on elapsed time."""
    if epochs_completed == 0:
        return None
    
    elapsed = time.time() - start_time
    time_per_epoch = elapsed / epochs_completed
    return time_per_epoch

def main():
    print("=" * 60)
    print("Training Monitor - Auto-Adjusting")
    print("=" * 60)
    
    start_time = time.time()
    check_interval = 120  # Check every 2 minutes
    max_time_per_epoch = 15 * 60  # 15 minutes max per epoch
    target_total_time = 90 * 60  # 90 minutes total target
    
    last_epoch_count = 0
    stall_count = 0
    
    while True:
        progress, status = check_training_progress()
        
        if progress is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
            if "not started" in status.lower():
                time.sleep(30)
                continue
            else:
                break
        
        epochs_completed = progress["epochs_completed"]
        elapsed = time.time() - start_time
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress Check:")
        print(f"  Epochs completed: {epochs_completed}/8")
        if progress["train_loss"]:
            print(f"  Latest train loss: {progress['train_loss']:.6f}")
        if progress["val_loss"]:
            print(f"  Latest val loss: {progress['val_loss']:.6f}")
        print(f"  Elapsed time: {elapsed/60:.1f} minutes")
        
        # Check if training is complete
        if epochs_completed >= 8:
            print("\n✓ Training complete!")
            break
        
        # Check if stalled
        if epochs_completed == last_epoch_count:
            stall_count += 1
            if stall_count >= 3:
                print("\n⚠ Training appears stalled. Checking...")
                time.sleep(60)
                progress2, _ = check_training_progress()
                if progress2 and progress2["epochs_completed"] == last_epoch_count:
                    print("⚠ Training stalled. Consider restarting with smaller model.")
        else:
            stall_count = 0
            last_epoch_count = epochs_completed
        
        # Estimate time remaining
        if epochs_completed > 0:
            time_per_epoch = elapsed / epochs_completed
            remaining_epochs = 8 - epochs_completed
            estimated_remaining = time_per_epoch * remaining_epochs
            total_estimated = elapsed + estimated_remaining
            
            print(f"  Time per epoch: {time_per_epoch/60:.1f} minutes")
            print(f"  Estimated remaining: {estimated_remaining/60:.1f} minutes")
            print(f"  Total estimated: {total_estimated/60:.1f} minutes")
            
            # Auto-adjust if too slow
            if time_per_epoch > max_time_per_epoch and epochs_completed < 3:
                print(f"\n⚠ Training too slow ({time_per_epoch/60:.1f} min/epoch)")
                print("⚠ Consider: Reduce epochs to 5-6 or make model smaller")
            
            if total_estimated > target_total_time * 1.2:  # 20% over target
                print(f"\n⚠ Estimated time ({total_estimated/60:.1f} min) exceeds target (90 min)")
                print("⚠ Recommendation: Stop and restart with 5-6 epochs")
        
        # Wait before next check
        time.sleep(check_interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

