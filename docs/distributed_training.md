# Distributed Training Guide

This guide explains how to set up and run distributed training for large language models using our framework.

## Overview

Distributed training allows you to train models across multiple GPUs and machines, enabling:

1. **Training larger models** that don't fit on a single GPU
2. **Faster training** by parallelizing computation
3. **More efficient resource utilization** in multi-GPU environments

Our framework supports several distributed training strategies:

- **DeepSpeed**: Zero Redundancy Optimizer (ZeRO) stages 1-3
- **FSDP**: Fully Sharded Data Parallel training
- **DDP**: Distributed Data Parallel for simpler setups

## Requirements

- Multiple GPUs (on one or more machines)
- Fast inter-GPU communication (NVLink or high-speed network for multi-node)
- CUDA-compatible GPUs with sufficient memory
- PyTorch 2.0+ and appropriate CUDA drivers

## Choosing a Distributed Strategy

| Strategy | Memory Efficiency | Training Speed | Setup Complexity | Best For |
|----------|------------------|----------------|------------------|----------|
| DeepSpeed ZeRO-1 | Good | Very Fast | Low | Multiple GPUs, medium models |
| DeepSpeed ZeRO-2 | Very Good | Fast | Medium | Limited GPU memory, larger models |
| DeepSpeed ZeRO-3 | Excellent | Moderate | High | Very large models, extreme memory constraints |
| FSDP | Very Good | Fast | Medium | PyTorch native solution, large models |
| DDP | Low | Very Fast | Low | Smaller models that fit on single GPU |

## DeepSpeed Configuration

DeepSpeed is a powerful deep learning optimization library that implements the Zero Redundancy Optimizer (ZeRO), which eliminates redundant storage of optimizer states and gradients.

### ZeRO Stages

- **Stage 1**: Shards optimizer states across GPUs
- **Stage 2**: Shards optimizer states and gradients
- **Stage 3**: Shards optimizer states, gradients, and model parameters

### Setting Up DeepSpeed

1. Create a DeepSpeed configuration:

```bash
python scripts/distributed_setup.py deepspeed --zero_stage 2 --offload_optimizer --bf16 --output config/ds_config.json
```

2. Modify your training configuration:

```yaml
distributed:
  use_deepspeed: true
  deepspeed_config: "config/ds_config.json"
  gradient_accumulation_steps: 16
```

### Example DeepSpeed Configuration

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 0.3,
  "zero_allow_untested_optimizer": true,
  "zero_force_ds_cpu_optimizer": false,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### Running with DeepSpeed

Generate a launcher script:

```bash
python scripts/distributed_setup.py launcher --config config.yaml --num_gpus_per_node 4 --use_deepspeed
```

Then run the generated script:

```bash
bash scripts/run_training.sh
```

## FSDP Configuration

Fully Sharded Data Parallel (FSDP) is PyTorch's native implementation of model sharding, similar to DeepSpeed ZeRO-3.

### Setting Up FSDP

1. Create an FSDP configuration:

```bash
python scripts/distributed_setup.py fsdp --sharding_strategy FULL_SHARD --mixed_precision BFLOAT16 --output config/fsdp_config.json
```

2. Modify your training configuration:

```yaml
distributed:
  use_fsdp: true
  fsdp_config: "config/fsdp_config.json"
  use_deepspeed: false
```

### Example FSDP Configuration

```json
{
  "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer", "MistralDecoderLayer", "GemmaDecoderLayer"],
  "fsdp_sharding_strategy": "FULL_SHARD",
  "fsdp_state_dict_type": "SHARDED_STATE_DICT",
  "fsdp_cuda_graphs": false,
  "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
  "fsdp_sync_module_states": true,
  "fsdp_use_orig_params": false,
  "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
  "fsdp_forward_prefetch": false,
  "fsdp_cpu_ram_efficient_loading": true,
  "fsdp_mixed_precision": "BFLOAT16"
}
```

### Running with FSDP

Generate a launcher script without DeepSpeed:

```bash
python scripts/distributed_setup.py launcher --config config.yaml --num_gpus_per_node 4
```

## Multi-Node Training

For training across multiple machines:

### Setup

1. Ensure all machines can communicate over the network
2. Set up password-less SSH between machines (for SLURM clusters)
3. Install the same environment on all machines

### SLURM Configuration

For clusters using SLURM, generate a SLURM script:

```bash
python scripts/distributed_setup.py slurm --config config.yaml --num_nodes 4 --num_gpus_per_node 8 --use_deepspeed --time_limit 48:00:00
```

This generates a script like:

```bash
#!/bin/bash

#SBATCH --job-name=llm_finetune
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Load required modules
module load cuda/12.2
module load anaconda3

# Activate conda environment
source activate llm_finetuning

# Set environment variables for better performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# Get master node information
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch training with DeepSpeed
srun deepspeed --num_gpus=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/train.py --config config.yaml --deepspeed config/ds_config.json
```

Submit the job:

```bash
sbatch scripts/slurm_job.sh
```

### Manual Multi-Node Setup

For manual setup (without SLURM):

1. Determine the master node's IP address
2. Generate launcher scripts for each node:

```bash
# On the master node (node_rank=0)
python scripts/distributed_setup.py launcher --config config.yaml --num_nodes 4 --num_gpus_per_node 8 --master_addr 192.168.1.100 --master_port 29500 --node_rank 0 --use_deepspeed

# On worker node 1 (node_rank=1)
python scripts/distributed_setup.py launcher --config config.yaml --num_nodes 4 --num_gpus_per_node 8 --master_addr 192.168.1.100 --master_port 29500 --node_rank 1 --use_deepspeed

# Repeat for each node with the appropriate node_rank
```

3. Run the launcher script on each node simultaneously

## Memory Optimization Techniques

### Gradient Checkpointing

Enable gradient checkpointing to trade computation for memory:

```yaml
training:
  gradient_checkpointing: true
```

### CPU Offloading

Offload parameters or optimizer states to CPU:

```json
"zero_optimization": {
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  },
  "offload_param": {
    "device": "cpu",
    "pin_memory": true
  }
}
```

### Mixed Precision

Use BF16 or FP16 to reduce memory usage:

```yaml
training:
  mixed_precision: "bf16"  # Options: "no", "fp16", "bf16"
```

## Troubleshooting

### Common Issues

#### 1. NCCL Errors

If you see NCCL communication errors:

```
# Enable NCCL debugging
export NCCL_DEBUG=INFO

# Disable IB if causing problems
export NCCL_IB_DISABLE=1

# Specify network interfaces to use
export NCCL_SOCKET_IFNAME=eth0
```

#### 2. Out of Memory (OOM) Errors

If you encounter OOM errors:

1. Reduce batch size (`micro_batch_size`) and increase gradient accumulation steps
2. Enable gradient checkpointing
3. Use a more aggressive optimization strategy (ZeRO-2/3)
4. Enable CPU offloading
5. Use mixed precision (BF16/FP16)

#### 3. Training Hangs

If training hangs:

1. Check network connectivity between nodes
2. Ensure firewall allows communication on the specified port
3. Increase timeout values for initialization

#### 4. Slow Training

If training is unexpectedly slow:

1. Check if network is bottlenecking (especially for ZeRO-3)
2. Verify NVLink is being used for multi-GPU communication
3. Optimize bucket sizes for communication

## Monitoring

Monitor your distributed training using:

1. **Tensorboard**: Enable with `report_to: "tensorboard"` in config
2. **Logging**: Check logs for performance metrics
3. **GPU Monitoring**: Use tools like `nvidia-smi` to monitor GPU usage

## Best Practices

1. **Start small**: Begin with a smaller model or subset of data to verify your setup
2. **Scale gradually**: Move from single GPU to multi-GPU, then to multi-node
3. **Checkpoint frequently**: Save checkpoints regularly in case of failures
4. **Test communication**: Verify network bandwidth between nodes before long runs
5. **Use the right strategy**: Match your distributed strategy to your hardware
6. **Optimize batch size**: Find the sweet spot between memory usage and training efficiency

## Advanced Topics

### Elastic Training

For environments where nodes may join or leave during training, consider implementing elastic training with PyTorch Elastic.

### Pipeline Parallelism

For extremely large models, combine ZeRO/FSDP with pipeline parallelism to distribute different layers across GPUs.

### Custom Communication Patterns

For specialized hardware setups, you may need to customize communication patterns by modifying the distributed configuration.
