#!/usr/bin/env python3
"""
Artemis Distributed Training Setup
==================================
This script configures distributed training for Artemis with DeepSpeed and FSDP
to enable efficient training on multiple GPUs and nodes.
"""

import os
import sys
import yaml
import json
import argparse
import logging
import subprocess
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def generate_deepspeed_config(
    config: Dict[str, Any], 
    output_path: str = "config/ds_config.json",
    zero_stage: int = 2,
) -> None:
    """
    Generate a DeepSpeed configuration file based on training settings.
    
    Args:
        config: Artemis configuration dictionary
        output_path: Path to save the DeepSpeed config
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
    """
    logger.info(f"Generating DeepSpeed configuration with ZeRO-{zero_stage}")
    
    training_config = config["training"]
    distributed_config = config.get("distributed", {})
    
    # Basic DeepSpeed configuration
    ds_config = {
        "train_batch_size": training_config["micro_batch_size"] * training_config["gradient_accumulation_steps"] * distributed_config.get("num_gpus_per_node", 1) * distributed_config.get("num_nodes", 1),
        "train_micro_batch_size_per_gpu": training_config["micro_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "steps_per_print": 100,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config["learning_rate"],
                "weight_decay": training_config["weight_decay"],
                "bias_correction": True,
            }
        },
        "scheduler": {
            "type": training_config["lr_scheduler_type"],
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_config["learning_rate"],
                "warmup_num_steps": int(training_config.get("warmup_ratio", 0.03) * (training_config["epochs"] * distributed_config.get("num_train_samples", 10000) / (training_config["micro_batch_size"] * training_config["gradient_accumulation_steps"] * distributed_config.get("num_gpus_per_node", 1) * distributed_config.get("num_nodes", 1)))),
            }
        },
        "fp16": {
            "enabled": "fp16" in training_config["mixed_precision"].lower(),
        },
        "bf16": {
            "enabled": "bf16" in training_config["mixed_precision"].lower(),
        },
        "gradient_clipping": training_config["max_grad_norm"],
        "zero_optimization": {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
    }
    
    # Add offload options for ZeRO-3
    if zero_stage >= 3:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
        if distributed_config.get("offload_parameters", False):
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }
    
    # Add activation checkpointing if enabled
    if training_config.get("gradient_checkpointing", False):
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": distributed_config.get("num_checkpoints", 1),
        }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save DeepSpeed configuration
    with open(output_path, "w") as f:
        json.dump(ds_config, f, indent=4)
    
    logger.info(f"DeepSpeed configuration saved to {output_path}")
    
    return output_path


def generate_launcher_script(
    config_path: str,
    num_gpus_per_node: int,
    num_nodes: int = 1,
    master_addr: str = "localhost",
    master_port: int = 29500,
    use_deepspeed: bool = True,
    use_fsdp: bool = False,
    node_rank: int = 0,
    output_path: str = "scripts/run_training.sh",
) -> None:
    """
    Generate a launcher script for distributed training.
    
    Args:
        config_path: Path to the configuration file
        num_gpus_per_node: Number of GPUs per node
        num_nodes: Number of nodes for distributed training
        master_addr: IP address of the master node
        master_port: Port for communication
        use_deepspeed: Whether to use DeepSpeed
        use_fsdp: Whether to use FSDP
        node_rank: Rank of this node (0 for master)
        output_path: Path to save the launcher script
    """
    logger.info(f"Generating launcher script for {num_nodes} nodes with {num_gpus_per_node} GPUs each")
    
    # Load the config to update distributed settings
    config = load_config(config_path)
    
    # Update distributed settings
    if "distributed" not in config:
        config["distributed"] = {}
    
    config["distributed"]["num_gpus_per_node"] = num_gpus_per_node
    config["distributed"]["num_nodes"] = num_nodes
    config["distributed"]["master_addr"] = master_addr
    config["distributed"]["master_port"] = master_port
    config["distributed"]["node_rank"] = node_rank
    
    # Set distribution method
    if use_deepspeed:
        config["distributed"]["use_deepspeed"] = True
        config["distributed"]["zero_stage"] = config["distributed"].get("zero_stage", 2)
        
        # Generate DeepSpeed config if it doesn't exist
        ds_config_path = config["distributed"].get("deepspeed_config", "config/ds_config.json")
        generate_deepspeed_config(
            config, 
            output_path=ds_config_path,
            zero_stage=config["distributed"]["zero_stage"]
        )
        config["distributed"]["deepspeed_config"] = ds_config_path
    
    if use_fsdp:
        config["distributed"]["use_fsdp"] = True
        # Add FSDP settings
        config["distributed"]["fsdp_sharding_strategy"] = config["distributed"].get("fsdp_sharding_strategy", "FULL_SHARD")
        config["distributed"]["fsdp_state_dict_type"] = config["distributed"].get("fsdp_state_dict_type", "SHARDED")
        config["distributed"]["fsdp_offload_params"] = config["distributed"].get("fsdp_offload_params", False)
    
    # Save updated config
    updated_config_path = "config/distributed_config.yaml"
    save_config(config, updated_config_path)
    
    # Create launcher script
    launcher_script = "#!/bin/bash\n\n"
    
    # Add DeepSpeed launcher
    if use_deepspeed:
        launcher_script += f"deepspeed --num_gpus={num_gpus_per_node} \\\n"
        if num_nodes > 1:
            launcher_script += f"  --num_nodes={num_nodes} \\\n"
            launcher_script += f"  --master_addr={master_addr} \\\n"
            launcher_script += f"  --master_port={master_port} \\\n"
            launcher_script += f"  --node_rank=$NODE_RANK \\\n"
        launcher_script += f"  src/train.py \\\n"
        launcher_script += f"  --config {updated_config_path}\n"
    
    # Add torchrun launcher for FSDP
    elif use_fsdp:
        launcher_script += f"torchrun \\\n"
        launcher_script += f"  --nnodes={num_nodes} \\\n"
        launcher_script += f"  --nproc_per_node={num_gpus_per_node} \\\n"
        if num_nodes > 1:
            launcher_script += f"  --master_addr={master_addr} \\\n"
            launcher_script += f"  --master_port={master_port} \\\n"
            launcher_script += f"  --node_rank=$NODE_RANK \\\n"
        launcher_script += f"  src/train.py \\\n"
        launcher_script += f"  --config {updated_config_path}\n"
    
    # Simple multi-GPU without DeepSpeed or FSDP
    else:
        launcher_script += f"CUDA_VISIBLE_DEVICES=0"
        for i in range(1, num_gpus_per_node):
            launcher_script += f",{i}"
        launcher_script += f" \\\n"
        launcher_script += f"python -m torch.distributed.launch \\\n"
        launcher_script += f"  --nproc_per_node={num_gpus_per_node} \\\n"
        if num_nodes > 1:
            launcher_script += f"  --nnodes={num_nodes} \\\n"
            launcher_script += f"  --master_addr={master_addr} \\\n"
            launcher_script += f"  --master_port={master_port} \\\n"
            launcher_script += f"  --node_rank=$NODE_RANK \\\n"
        launcher_script += f"  src/train.py \\\n"
        launcher_script += f"  --config {updated_config_path}\n"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write launcher script
    with open(output_path, "w") as f:
        f.write(launcher_script)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    logger.info(f"Launcher script saved to {output_path}")


def launch_training(
    launcher_script: str,
    num_nodes: int = 1,
) -> None:
    """
    Launch training using the generated launcher script.
    
    Args:
        launcher_script: Path to the launcher script
        num_nodes: Number of nodes for distributed training
    """
    logger.info(f"Launching training on {num_nodes} nodes")
    
    # Single node training
    if num_nodes == 1:
        logger.info("Starting single-node training")
        subprocess.run(f"bash {launcher_script}", shell=True, check=True)
    
    # Multi-node training (would typically be handled by job scheduler)
    else:
        logger.info("Multi-node training requires a job scheduler or manual setup on each node")
        logger.info(f"Please run 'NODE_RANK=<rank> bash {launcher_script}' on each node")
        
        # On master node (rank 0)
        logger.info("Starting training on master node (rank 0)")
        subprocess.run(f"NODE_RANK=0 bash {launcher_script}", shell=True, check=True)


def main():
    """Parse arguments and set up distributed training."""
    parser = argparse.ArgumentParser(description="Artemis Distributed Training Setup")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup parser for generating DeepSpeed config
    ds_parser = subparsers.add_parser("ds_config", help="Generate DeepSpeed configuration")
    ds_parser.add_argument("--config", type=str, required=True, help="Path to Artemis configuration file")
    ds_parser.add_argument("--output", type=str, default="config/ds_config.json", help="Output path for DeepSpeed config")
    ds_parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="ZeRO optimization stage")
    
    # Setup parser for generating launcher script
    launcher_parser = subparsers.add_parser("launcher", help="Generate launcher script for distributed training")
    launcher_parser.add_argument("--config", type=str, required=True, help="Path to Artemis configuration file")
    launcher_parser.add_argument("--num_gpus_per_node", type=int, default=1, help="Number of GPUs per node")
    launcher_parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    launcher_parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    launcher_parser.add_argument("--master_port", type=int, default=29500, help="Master node port")
    launcher_parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    launcher_parser.add_argument("--use_fsdp", action="store_true", help="Use FSDP")
    launcher_parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node")
    launcher_parser.add_argument("--output", type=str, default="scripts/run_training.sh", help="Output path for launcher script")
    
    # Setup parser for launching training
    launch_parser = subparsers.add_parser("launch", help="Launch distributed training")
    launch_parser.add_argument("--script", type=str, default="scripts/run_training.sh", help="Path to launcher script")
    launch_parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "ds_config":
        config = load_config(args.config)
        generate_deepspeed_config(config, args.output, args.zero_stage)
    
    elif args.command == "launcher":
        generate_launcher_script(
            args.config,
            args.num_gpus_per_node,
            args.num_nodes,
            args.master_addr,
            args.master_port,
            args.use_deepspeed,
            args.use_fsdp,
            args.node_rank,
            args.output
        )
    
    elif args.command == "launch":
        launch_training(args.script, args.num_nodes)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
