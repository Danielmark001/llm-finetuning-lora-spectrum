#!/usr/bin/env python3
"""
Distributed Training Setup for LLM Fine-Tuning
==============================================
This script sets up distributed training environments for LLM fine-tuning,
including DeepSpeed, FSDP, and multi-node configurations.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from typing import Dict, Optional

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


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_deepspeed_config(
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 0.3,
    fp16_enabled: bool = False,
    bf16_enabled: bool = True,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Generate a DeepSpeed configuration file.
    
    Args:
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU/NVMe
        offload_param: Whether to offload parameters to CPU/NVMe
        gradient_accumulation_steps: Number of gradient accumulation steps
        gradient_clipping: Gradient clipping threshold
        fp16_enabled: Whether to enable fp16 precision
        bf16_enabled: Whether to enable bf16 precision
        output_path: Path to save the configuration
        
    Returns:
        Dict: DeepSpeed configuration
    """
    logger.info(f"Generating DeepSpeed configuration with ZeRO-{zero_stage}")
    
    # Base configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
    }
    
    # Configure precision
    if bf16_enabled:
        ds_config["bf16"] = {
            "enabled": True
        }
    elif fp16_enabled:
        ds_config["fp16"] = {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    
    # Configure ZeRO stage
    if zero_stage > 0:
        ds_config["zero_optimization"] = {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
        
        # Add offloading configuration for ZeRO-2 and ZeRO-3
        if zero_stage >= 2 and offload_optimizer:
            if "zero_optimization" not in ds_config:
                ds_config["zero_optimization"] = {}
            
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        
        # Parameter offloading only available for ZeRO-3
        if zero_stage == 3 and offload_param:
            if "zero_optimization" not in ds_config:
                ds_config["zero_optimization"] = {}
            
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }
    
    # Save configuration if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ds_config, f, indent=2)
        logger.info(f"DeepSpeed configuration saved to {output_path}")
    
    return ds_config


def generate_fsdp_config(
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: str = "BFLOAT16",
    checkpoint_strategy: str = "SHARDED_CHECKPOINT",
    output_path: Optional[str] = None,
) -> Dict:
    """
    Generate a FSDP (Fully Sharded Data Parallel) configuration.
    
    Args:
        sharding_strategy: Sharding strategy (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD)
        mixed_precision: Mixed precision mode (FULL, BFLOAT16, FP16)
        checkpoint_strategy: Checkpoint strategy (FULL, SHARDED, LOCAL)
        output_path: Path to save the configuration
        
    Returns:
        Dict: FSDP configuration
    """
    logger.info(f"Generating FSDP configuration with {sharding_strategy} strategy")
    
    # Map string options to corresponding values
    sharding_strategies = {
        "FULL_SHARD": "FULL_SHARD",  # Full sharding of model parameters, gradients, and optimizer states
        "SHARD_GRAD_OP": "SHARD_GRAD_OP",  # Shard gradients and optimizer states only
        "NO_SHARD": "NO_SHARD",  # No sharding, similar to DDP
    }
    
    mixed_precision_modes = {
        "FULL": "FULL",  # Use full precision
        "BFLOAT16": "BFLOAT16",  # Use BFloat16 precision
        "FP16": "FP16",  # Use FP16 precision
    }
    
    checkpoint_strategies = {
        "FULL_CHECKPOINT": "FULL_STATE_DICT",  # Full model checkpointing
        "SHARDED_CHECKPOINT": "SHARDED_STATE_DICT",  # Sharded checkpointing
        "LOCAL_CHECKPOINT": "LOCAL_STATE_DICT",  # Local checkpointing
    }
    
    # Create configuration
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer", "MistralDecoderLayer", "GemmaDecoderLayer"],
        "fsdp_sharding_strategy": sharding_strategies.get(sharding_strategy, "FULL_SHARD"),
        "fsdp_state_dict_type": checkpoint_strategies.get(checkpoint_strategy, "SHARDED_STATE_DICT"),
        "fsdp_cuda_graphs": False,
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_sync_module_states": True,
        "fsdp_use_orig_params": False,
        "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
        "fsdp_forward_prefetch": False,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_mixed_precision": mixed_precision_modes.get(mixed_precision, "BFLOAT16"),
    }
    
    # Save configuration if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(fsdp_config, f, indent=2)
        logger.info(f"FSDP configuration saved to {output_path}")
    
    return fsdp_config


def generate_launcher_script(
    config_path: str,
    output_path: str,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    master_addr: str = "localhost",
    master_port: int = 29500,
    use_deepspeed: bool = True,
    node_rank: int = 0,
):
    """
    Generate a launcher script for distributed training.
    
    Args:
        config_path: Path to the training configuration
        output_path: Path to save the launcher script
        num_nodes: Number of nodes
        num_gpus_per_node: Number of GPUs per node
        master_addr: Master node address
        master_port: Master node port
        use_deepspeed: Whether to use DeepSpeed
        node_rank: Rank of this node in multi-node setup
    """
    logger.info(f"Generating launcher script for {num_nodes} nodes with {num_gpus_per_node} GPUs each")
    
    # Load configuration to determine if DeepSpeed or FSDP should be used
    config = load_config(config_path)
    
    # Override based on function parameter
    use_distributed = num_nodes > 1 or num_gpus_per_node > 1
    
    # Generate appropriate command
    if use_distributed:
        if use_deepspeed:
            # DeepSpeed command
            cmd = (
                f"deepspeed --num_nodes={num_nodes} --num_gpus={num_gpus_per_node} "
                f"--master_addr={master_addr} --master_port={master_port} "
            )
            
            if num_nodes > 1:
                cmd += f"--node_rank={node_rank} "
            
            ds_config_path = "config/ds_config.json"
            if not os.path.exists(ds_config_path):
                # Generate DeepSpeed config if not exists
                ds_config = generate_deepspeed_config(
                    zero_stage=config["distributed"].get("zero_stage", 2),
                    offload_optimizer=config["distributed"].get("offload_optimizer", False),
                    offload_param=config["distributed"].get("offload_param", False),
                    gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
                    gradient_clipping=config["training"].get("max_grad_norm", 0.3),
                    fp16_enabled="fp16" in config["training"].get("mixed_precision", "").lower(),
                    bf16_enabled="bf16" in config["training"].get("mixed_precision", "").lower(),
                    output_path=ds_config_path,
                )
            
            cmd += f"src/train.py --config {config_path} --deepspeed {ds_config_path}"
        
        else:
            # torch.distributed.launch command
            cmd = (
                f"torchrun --nnodes={num_nodes} --nproc_per_node={num_gpus_per_node} "
                f"--master_addr={master_addr} --master_port={master_port} "
            )
            
            if num_nodes > 1:
                cmd += f"--node_rank={node_rank} "
            
            cmd += f"src/train.py --config {config_path}"
            
            # Check if FSDP should be used
            if config["distributed"].get("use_fsdp", False):
                fsdp_config_path = "config/fsdp_config.json"
                if not os.path.exists(fsdp_config_path):
                    # Generate FSDP config if not exists
                    fsdp_config = generate_fsdp_config(
                        sharding_strategy=config["distributed"].get("fsdp_sharding_strategy", "FULL_SHARD"),
                        mixed_precision=config["training"].get("mixed_precision", "BFLOAT16").upper(),
                        checkpoint_strategy=config["distributed"].get("fsdp_checkpoint_strategy", "SHARDED_CHECKPOINT"),
                        output_path=fsdp_config_path,
                    )
                
                cmd += f" --fsdp_config {fsdp_config_path}"
    
    else:
        # Single GPU command
        cmd = f"python src/train.py --config {config_path}"
    
    # Add environment variables for better performance
    env_vars = [
        "export CUDA_DEVICE_MAX_CONNECTIONS=1",
        "export NCCL_ASYNC_ERROR_HANDLING=1",
        "export NCCL_DEBUG=INFO",
    ]
    
    # Assemble the script
    script_content = "#!/bin/bash\n\n"
    script_content += "# Generated launcher script for distributed training\n\n"
    
    # Add environment variables
    script_content += "# Set environment variables for better performance\n"
    for var in env_vars:
        script_content += f"{var}\n"
    
    script_content += "\n# Launch training\n"
    script_content += f"{cmd}\n"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write script to file
    with open(output_path, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_path, 0o755)
    
    logger.info(f"Launcher script generated at {output_path}")


def generate_slurm_job_script(
    config_path: str,
    output_path: str,
    job_name: str = "llm_finetune",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    time_limit: str = "24:00:00",
    partition: str = "gpu",
    use_deepspeed: bool = True,
    memory_per_gpu: str = "32G",
    output_log: str = "slurm-%j.out",
):
    """
    Generate a SLURM job script for distributed training.
    
    Args:
        config_path: Path to the training configuration
        output_path: Path to save the job script
        job_name: SLURM job name
        num_nodes: Number of nodes
        num_gpus_per_node: Number of GPUs per node
        time_limit: Time limit for the job
        partition: SLURM partition
        use_deepspeed: Whether to use DeepSpeed
        memory_per_gpu: Memory per GPU
        output_log: Output log file pattern
    """
    logger.info(f"Generating SLURM job script for {num_nodes} nodes with {num_gpus_per_node} GPUs each")
    
    # Load configuration to determine if DeepSpeed or FSDP should be used
    config = load_config(config_path)
    
    # SLURM header
    script_content = "#!/bin/bash\n\n"
    script_content += f"#SBATCH --job-name={job_name}\n"
    script_content += f"#SBATCH --nodes={num_nodes}\n"
    script_content += f"#SBATCH --ntasks-per-node={num_gpus_per_node}\n"
    script_content += f"#SBATCH --gres=gpu:{num_gpus_per_node}\n"
    script_content += f"#SBATCH --cpus-per-task=8\n"
    script_content += f"#SBATCH --mem-per-gpu={memory_per_gpu}\n"
    script_content += f"#SBATCH --time={time_limit}\n"
    script_content += f"#SBATCH --partition={partition}\n"
    script_content += f"#SBATCH --output={output_log}\n"
    script_content += f"#SBATCH --error={output_log}\n\n"
    
    # Environment setup
    script_content += "# Load required modules (adjust as needed for your cluster)\n"
    script_content += "module load cuda/12.2\n"
    script_content += "module load anaconda3\n\n"
    
    script_content += "# Activate conda environment\n"
    script_content += "source activate llm_finetuning\n\n"
    
    # Environment variables for better performance
    script_content += "# Set environment variables for better performance\n"
    script_content += "export CUDA_DEVICE_MAX_CONNECTIONS=1\n"
    script_content += "export NCCL_ASYNC_ERROR_HANDLING=1\n"
    script_content += "export NCCL_DEBUG=INFO\n"
    script_content += "export NCCL_IB_DISABLE=0\n"
    script_content += "export NCCL_SOCKET_IFNAME=^lo,docker\n\n"
    
    # Master node information
    script_content += "# Get master node information\n"
    script_content += "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)\n"
    script_content += "export MASTER_PORT=29500\n\n"
    
    # For multi-node jobs, use SLURM_PROCID as node rank
    if num_nodes > 1:
        script_content += "# Use SLURM_PROCID as node rank\n"
        script_content += "export NODE_RANK=$SLURM_PROCID\n\n"
    
    # Generate training command
    if use_deepspeed:
        # DeepSpeed command
        script_content += "# Launch training with DeepSpeed\n"
        
        ds_config_path = "config/ds_config.json"
        if not os.path.exists(ds_config_path):
            # Generate DeepSpeed config if not exists
            ds_config = generate_deepspeed_config(
                zero_stage=config["distributed"].get("zero_stage", 2),
                offload_optimizer=config["distributed"].get("offload_optimizer", False),
                offload_param=config["distributed"].get("offload_param", False),
                gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
                gradient_clipping=config["training"].get("max_grad_norm", 0.3),
                fp16_enabled="fp16" in config["training"].get("mixed_precision", "").lower(),
                bf16_enabled="bf16" in config["training"].get("mixed_precision", "").lower(),
                output_path=ds_config_path,
            )
        
        script_content += (
            f"srun deepspeed --num_gpus={num_gpus_per_node} "
            f"--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
        )
        
        if num_nodes > 1:
            script_content += "--node_rank=$NODE_RANK "
        
        script_content += f"src/train.py --config {config_path} --deepspeed {ds_config_path}\n"
    
    else:
        # torchrun command
        script_content += "# Launch training with torchrun\n"
        
        script_content += (
            f"srun torchrun --nnodes={num_nodes} --nproc_per_node={num_gpus_per_node} "
            f"--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
        )
        
        if num_nodes > 1:
            script_content += "--node_rank=$NODE_RANK "
        
        script_content += f"src/train.py --config {config_path}"
        
        # Check if FSDP should be used
        if config["distributed"].get("use_fsdp", False):
            fsdp_config_path = "config/fsdp_config.json"
            if not os.path.exists(fsdp_config_path):
                # Generate FSDP config if not exists
                fsdp_config = generate_fsdp_config(
                    sharding_strategy=config["distributed"].get("fsdp_sharding_strategy", "FULL_SHARD"),
                    mixed_precision=config["training"].get("mixed_precision", "BFLOAT16").upper(),
                    checkpoint_strategy=config["distributed"].get("fsdp_checkpoint_strategy", "SHARDED_CHECKPOINT"),
                    output_path=fsdp_config_path,
                )
            
            script_content += f" --fsdp_config {fsdp_config_path}"
        
        script_content += "\n"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write script to file
    with open(output_path, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_path, 0o755)
    
    logger.info(f"SLURM job script generated at {output_path}")


def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(description="Distributed Training Setup")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # DeepSpeed config command
    ds_parser = subparsers.add_parser("deepspeed", help="Generate DeepSpeed configuration")
    ds_parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="ZeRO optimization stage")
    ds_parser.add_argument("--offload_optimizer", action="store_true", help="Offload optimizer states to CPU/NVMe")
    ds_parser.add_argument("--offload_param", action="store_true", help="Offload parameters to CPU/NVMe")
    ds_parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    ds_parser.add_argument("--gradient_clipping", type=float, default=0.3, help="Gradient clipping threshold")
    ds_parser.add_argument("--fp16", action="store_true", help="Enable fp16 precision")
    ds_parser.add_argument("--bf16", action="store_true", help="Enable bf16 precision")
    ds_parser.add_argument("--output", type=str, default="config/ds_config.json", help="Output path for the configuration")
    
    # FSDP config command
    fsdp_parser = subparsers.add_parser("fsdp", help="Generate FSDP configuration")
    fsdp_parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD", 
                            choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                            help="Sharding strategy")
    fsdp_parser.add_argument("--mixed_precision", type=str, default="BFLOAT16",
                            choices=["FULL", "BFLOAT16", "FP16"],
                            help="Mixed precision mode")
    fsdp_parser.add_argument("--checkpoint_strategy", type=str, default="SHARDED_CHECKPOINT",
                            choices=["FULL_CHECKPOINT", "SHARDED_CHECKPOINT", "LOCAL_CHECKPOINT"],
                            help="Checkpoint strategy")
    fsdp_parser.add_argument("--output", type=str, default="config/fsdp_config.json", help="Output path for the configuration")
    
    # Launcher script command
    launcher_parser = subparsers.add_parser("launcher", help="Generate launcher script")
    launcher_parser.add_argument("--config", type=str, required=True, help="Path to the training configuration")
    launcher_parser.add_argument("--output", type=str, default="scripts/run_training.sh", help="Output path for the script")
    launcher_parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    launcher_parser.add_argument("--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node")
    launcher_parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    launcher_parser.add_argument("--master_port", type=int, default=29500, help="Master node port")
    launcher_parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    launcher_parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node in multi-node setup")
    
    # SLURM job script command
    slurm_parser = subparsers.add_parser("slurm", help="Generate SLURM job script")
    slurm_parser.add_argument("--config", type=str, required=True, help="Path to the training configuration")
    slurm_parser.add_argument("--output", type=str, default="scripts/slurm_job.sh", help="Output path for the script")
    slurm_parser.add_argument("--job_name", type=str, default="llm_finetune", help="SLURM job name")
    slurm_parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    slurm_parser.add_argument("--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node")
    slurm_parser.add_argument("--time_limit", type=str, default="24:00:00", help="Time limit for the job")
    slurm_parser.add_argument("--partition", type=str, default="gpu", help="SLURM partition")
    slurm_parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    slurm_parser.add_argument("--memory_per_gpu", type=str, default="32G", help="Memory per GPU")
    slurm_parser.add_argument("--output_log", type=str, default="slurm-%j.out", help="Output log file pattern")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "deepspeed":
        generate_deepspeed_config(
            zero_stage=args.zero_stage,
            offload_optimizer=args.offload_optimizer,
            offload_param=args.offload_param,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.gradient_clipping,
            fp16_enabled=args.fp16,
            bf16_enabled=args.bf16,
            output_path=args.output,
        )
    
    elif args.command == "fsdp":
        generate_fsdp_config(
            sharding_strategy=args.sharding_strategy,
            mixed_precision=args.mixed_precision,
            checkpoint_strategy=args.checkpoint_strategy,
            output_path=args.output,
        )
    
    elif args.command == "launcher":
        generate_launcher_script(
            config_path=args.config,
            output_path=args.output,
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.num_gpus_per_node,
            master_addr=args.master_addr,
            master_port=args.master_port,
            use_deepspeed=args.use_deepspeed,
            node_rank=args.node_rank,
        )
    
    elif args.command == "slurm":
        generate_slurm_job_script(
            config_path=args.config,
            output_path=args.output,
            job_name=args.job_name,
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.num_gpus_per_node,
            time_limit=args.time_limit,
            partition=args.partition,
            use_deepspeed=args.use_deepspeed,
            memory_per_gpu=args.memory_per_gpu,
            output_log=args.output_log,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
