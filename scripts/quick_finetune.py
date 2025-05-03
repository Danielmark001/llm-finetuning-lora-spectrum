#!/usr/bin/env python3
"""
Quick Fine-tuning Script
========================
This script provides a simplified interface for fine-tuning language models
using the framework. It offers sensible defaults and minimal configuration.

Example:
    python quick_finetune.py --model_name "meta-llama/Llama-3.1-8B-Instruct" \
                            --dataset_path "data/train.json" \
                            --method "qlora" \
                            --output_dir "models/quick_finetune"
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
import subprocess
from typing import Dict, Optional, List, Union

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def detect_dataset_format(dataset_path: str) -> str:
    """
    Detect the format of the dataset based on its content.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        str: Detected format ("alpaca", "sharegpt", or "unknown")
    """
    try:
        with open(dataset_path, "r") as f:
            # Read the first sample
            data = json.load(f)
            
            if not data:
                return "unknown"
            
            # Get the first example
            sample = data[0]
            
            # Check for Alpaca format
            if all(key in sample for key in ["instruction", "output"]):
                return "alpaca"
            
            # Check for ShareGPT format
            if "conversations" in sample:
                return "sharegpt"
            
            return "unknown"
    except Exception as e:
        logger.warning(f"Error detecting dataset format: {e}")
        return "unknown"


def detect_gpu_count() -> int:
    """
    Detect the number of available GPUs.
    
    Returns:
        int: Number of GPUs available
    """
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        logger.warning("Could not detect GPUs with torch. Assuming 1 GPU.")
        return 1


def generate_config(
    model_name: str,
    dataset_path: str,
    eval_path: Optional[str],
    method: str,
    output_dir: str,
    learning_rate: float,
    epochs: int,
    micro_batch_size: int,
    max_seq_length: int,
    gradient_accumulation_steps: int,
) -> Dict:
    """
    Generate a configuration dict for fine-tuning.
    
    Args:
        model_name: Name or path of the model
        dataset_path: Path to the dataset
        eval_path: Path to the evaluation dataset
        method: Fine-tuning method
        output_dir: Directory to save the outputs
        learning_rate: Learning rate
        epochs: Number of epochs
        micro_batch_size: Batch size per GPU
        max_seq_length: Maximum sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        Dict: Configuration dictionary
    """
    # Detect dataset format
    dataset_format = detect_dataset_format(dataset_path)
    if dataset_format == "unknown":
        logger.warning("Could not detect dataset format. Defaulting to alpaca.")
        dataset_format = "alpaca"
    
    # Detect number of GPUs
    num_gpus = detect_gpu_count()
    use_distributed = num_gpus > 1
    
    # Set appropriate target modules based on model name
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "falcon" in model_name.lower():
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "mistral" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "llama" in model_name.lower() or "gemma" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Create configuration
    config = {
        "model": {
            "base_model": model_name,
            "tokenizer": model_name,
            "load_in_8bit": method == "lora",
            "load_in_4bit": method == "qlora",
            "trust_remote_code": True,
            "use_flash_attention": True,
        },
        "fine_tuning": {
            "method": method,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": target_modules,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "spectrum": {
                "snr_threshold": 0.5,
                "layers_to_finetune": "auto",
            },
            "quantization": {
                "bits": 4,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        },
        "training": {
            "epochs": epochs,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.3,
            "optimizer": "paged_adamw_8bit",
            "weight_decay": 0.001,
            "max_seq_length": max_seq_length,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
        },
        "dataset": {
            "format": dataset_format,
            "train_path": dataset_path,
            "eval_path": eval_path,
            "preprocessing": {
                "add_eos_token": True,
                "add_bos_token": False,
                "use_chat_template": True,
            },
        },
        "output": {
            "output_dir": output_dir,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 100,
            "save_total_limit": 5,
            "push_to_hub": False,
            "hub_model_id": None,
        },
        "evaluation": {
            "do_eval": eval_path is not None,
            "eval_batch_size": 8,
            "eval_strategy": "steps",
            "eval_steps": 100,
        },
    }
    
    # Add distributed training configuration if multiple GPUs
    if use_distributed:
        # Create DeepSpeed config directory if it doesn't exist
        os.makedirs(os.path.join(project_root, "config"), exist_ok=True)
        ds_config_path = os.path.join(project_root, "config", "ds_config.json")
        
        # Create a simple DeepSpeed config
        ds_config = {
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": 0.3,
            "zero_allow_untested_optimizer": True,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            }
        }
        
        # Save DeepSpeed config
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)
        
        # Add distributed config to main config
        config["distributed"] = {
            "use_deepspeed": True,
            "deepspeed_config": ds_config_path,
            "zero_stage": 2,
        }
    
    return config


def main():
    """Main function to handle CLI arguments and run fine-tuning."""
    parser = argparse.ArgumentParser(description="Quick Fine-tuning Script")
    
    # Required arguments
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the outputs")
    
    # Optional arguments
    parser.add_argument("--eval_path", type=str, help="Path to the evaluation dataset")
    parser.add_argument("--method", type=str, default="qlora", choices=["full", "lora", "qlora", "spectrum"], help="Fine-tuning method")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--save_config_only", action="store_true", help="Only save the config file without training")
    
    args = parser.parse_args()
    
    # Generate configuration
    config = generate_config(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        eval_path=args.eval_path,
        method=args.method,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    logger.info(f"Configuration saved to {config_path}")
    
    # Run training if not save_config_only
    if not args.save_config_only:
        train_script = os.path.join(project_root, "src", "train.py")
        
        # Check if we need to use distributed training
        num_gpus = detect_gpu_count()
        
        if num_gpus > 1 and config.get("distributed", {}).get("use_deepspeed", False):
            # Use DeepSpeed for multi-GPU training
            command = [
                "deepspeed",
                f"--num_gpus={num_gpus}",
                train_script,
                "--config", config_path,
            ]
        else:
            # Use single GPU training
            command = [
                "python",
                train_script,
                "--config", config_path,
            ]
        
        logger.info(f"Starting training with command: {' '.join(command)}")
        
        # Run the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end="")
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info("Training completed successfully!")
        else:
            logger.error(f"Training failed with return code {process.returncode}")
    else:
        logger.info("Configuration saved. Use the following command to start training:")
        logger.info(f"python {os.path.join(project_root, 'src', 'train.py')} --config {config_path}")


if __name__ == "__main__":
    main()
