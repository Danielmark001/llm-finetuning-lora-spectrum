# Configuration Guide

This document explains how to configure the fine-tuning framework for different models, datasets, and training approaches.

## Configuration File

The framework uses a YAML configuration file to specify all aspects of the fine-tuning process. Here's a comprehensive example:

```yaml
# Model settings
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
  load_in_8bit: false
  load_in_4bit: true
  trust_remote_code: true
  use_flash_attention: true
  use_liger_kernels: false  # Advanced optimization

# Fine-tuning method
fine_tuning:
  method: "qlora"  # Options: "full", "lora", "qlora", "spectrum"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias: "none"
    task_type: "CAUSAL_LM"
  spectrum:
    snr_threshold: 0.5
    layers_to_finetune: "auto"  # Will be determined by SNR analysis
  quantization:
    bits: 4
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true

# Training parameters
training:
  epochs: 3
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  max_grad_norm: 0.3
  optimizer: "paged_adamw_8bit"
  weight_decay: 0.001
  max_seq_length: 4096
  gradient_checkpointing: true
  mixed_precision: "bf16"  # Options: "no", "fp16", "bf16"

# Dataset configuration
dataset:
  format: "alpaca"  # Options: "alpaca", "sharegpt", "oasst", "custom"
  train_path: "data/train.json"
  eval_path: "data/eval.json"
  preprocessing:
    add_eos_token: true
    add_bos_token: false
    use_chat_template: true

# Distributed training
distributed:
  use_deepspeed: true
  deepspeed_config: "config/ds_config.json"
  zero_stage: 2
  gradient_accumulation_steps: 16

# Logging and checkpoints
output:
  output_dir: "models/runs"
  logging_steps: 10
  eval_steps: 100
  save_steps: 500
  save_total_limit: 5
  push_to_hub: false
  hub_model_id: null

# Evaluation settings
evaluation:
  do_eval: true
  eval_batch_size: 8
  eval_strategy: "steps"
  eval_steps: 200
  benchmarks:
    - "lm-evaluation-harness"
    - "domain-specific-eval"
    - "human-eval"
```

## Model Configuration

The `model` section configures the base model and loading options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_model` | HuggingFace model ID or local path | Required |
| `tokenizer` | Tokenizer to use (usually same as model) | Same as `base_model` |
| `load_in_8bit` | Whether to load in 8-bit quantization | `false` |
| `load_in_4bit` | Whether to load in 4-bit quantization | `false` |
| `trust_remote_code` | Whether to trust remote code in model repos | `true` |
| `use_flash_attention` | Whether to use Flash Attention 2 | `true` |
| `use_liger_kernels` | Whether to use Liger kernels for attention | `false` |

## Fine-tuning Method Configuration

The `fine_tuning` section specifies which method to use and its parameters:

### Method Options

| Method | Description | Best For |
|--------|-------------|----------|
| `full` | Full parameter fine-tuning | When you have significant compute resources |
| `lora` | Low-Rank Adaptation | Balance between efficiency and performance |
| `qlora` | Quantized Low-Rank Adaptation | Limited GPU memory |
| `spectrum` | Signal-to-Noise Ratio based selection | When you want to target specific model capabilities |

### LoRA / QLoRA Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `r` | Rank of the adaptation matrices | 8-32 (higher = more capacity) |
| `alpha` | Scaling factor | Typically 2x the rank |
| `dropout` | Dropout rate for regularization | 0.05-0.1 |
| `target_modules` | Which modules to apply LoRA to | Model-dependent |
| `bias` | Whether to train bias parameters | `"none"`, `"all"`, `"lora_only"` |

### Spectrum Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `snr_threshold` | SNR percentile threshold (0.0 to 1.0) | 0.5-0.7 (higher = fewer layers) |
| `layers_to_finetune` | `"auto"` or list of layer indices | `"auto"` for automatic selection |

### Quantization Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `bits` | Bit precision for quantization | 4 or 8 |
| `bnb_4bit_compute_dtype` | Compute dtype for 4-bit quantization | `"bfloat16"` or `"float16"` |
| `bnb_4bit_quant_type` | Quantization type | `"nf4"` or `"fp4"` |
| `bnb_4bit_use_double_quant` | Whether to use double quantization | `true` for extra memory savings |

## Training Parameters

The `training` section controls the training process:

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `epochs` | Number of training epochs | 2-5 for most cases |
| `micro_batch_size` | Batch size per GPU | 1-8 depending on GPU memory |
| `gradient_accumulation_steps` | Steps before optimizer update | 8-32 for effective larger batch |
| `learning_rate` | Learning rate | 1e-5 to 3e-4 |
| `lr_scheduler_type` | Learning rate scheduler | `"cosine"`, `"linear"`, `"constant_with_warmup"` |
| `warmup_ratio` | Fraction of steps for warmup | 0.03-0.1 |
| `max_grad_norm` | Gradient clipping threshold | 0.3-1.0 |
| `optimizer` | Optimizer to use | `"adamw_8bit"`, `"paged_adamw_8bit"`, `"adamw_torch"` |
| `weight_decay` | L2 regularization strength | 0.001-0.01 |
| `max_seq_length` | Maximum sequence length | Model-dependent (typically 2048-8192) |
| `gradient_checkpointing` | Trade compute for memory | `true` for large models |
| `mixed_precision` | Mixed precision training | `"bf16"`, `"fp16"`, or `"no"` |

## Dataset Configuration

The `dataset` section configures the training data:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `format` | Dataset format | `"alpaca"`, `"sharegpt"`, `"oasst"`, `"custom"` |
| `train_path` | Path to training data | Path to JSON file |
| `eval_path` | Path to evaluation data | Path to JSON file |
| `preprocessing.add_eos_token` | Add EOS token | `true`/`false` |
| `preprocessing.add_bos_token` | Add BOS token | `true`/`false` |
| `preprocessing.use_chat_template` | Use tokenizer's chat template | `true`/`false` |

## Distributed Training

The `distributed` section configures distributed training:

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `use_deepspeed` | Whether to use DeepSpeed | `true` for multi-GPU training |
| `deepspeed_config` | Path to DeepSpeed config | Path to JSON file |
| `zero_stage` | ZeRO optimization stage | 1-3 (higher = more memory efficient) |
| `gradient_accumulation_steps` | Steps before optimizer update | 16-64 for distributed training |

## Output Configuration

The `output` section controls logging and model saving:

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `output_dir` | Directory to save models | Path to directory |
| `logging_steps` | Steps between logging | 10-50 |
| `eval_steps` | Steps between evaluations | 100-500 |
| `save_steps` | Steps between saving checkpoints | 100-1000 |
| `save_total_limit` | Maximum number of checkpoints | 3-10 |
| `push_to_hub` | Whether to upload to HF Hub | `true`/`false` |
| `hub_model_id` | Model ID for HF Hub | `"username/model-name"` |

## Evaluation Settings

The `evaluation` section configures model evaluation:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `do_eval` | Whether to evaluate during training | `true`/`false` |
| `eval_batch_size` | Batch size for evaluation | 8-32 |
| `eval_strategy` | When to evaluate | `"steps"`, `"epoch"`, `"no"` |
| `eval_steps` | Steps between evaluations | 100-500 |
| `benchmarks` | Benchmarks to run | List of benchmark names |

## Configuration Templates

### QLoRA for Consumer GPUs (16GB VRAM)

```yaml
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  load_in_4bit: true
  use_flash_attention: true

fine_tuning:
  method: "qlora"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  epochs: 3
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  max_seq_length: 2048
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

### Spectrum for Targeted Fine-tuning

```yaml
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  load_in_4bit: true

fine_tuning:
  method: "spectrum"
  spectrum:
    snr_threshold: 0.5
    layers_to_finetune: "auto"

training:
  epochs: 3
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.5e-4
  gradient_checkpointing: true
```

### Multi-GPU Full Fine-tuning

```yaml
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  use_flash_attention: true

fine_tuning:
  method: "full"

training:
  epochs: 2
  micro_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  gradient_checkpointing: true
  mixed_precision: "bf16"

distributed:
  use_deepspeed: true
  deepspeed_config: "config/ds_config.json"
  zero_stage: 3
```

## Command-line Usage

To use a configuration file with the training script:

```bash
python src/train.py --config config.yaml
```

You can also override specific configuration values from the command line:

```bash
python src/train.py --config config.yaml --model.base_model "mistralai/Mistral-7B-Instruct-v0.2" --training.learning_rate 1e-4
```

## Creating Custom Configurations

To create a custom configuration:

1. Start with one of the templates above
2. Adjust the parameters for your specific use case
3. Save the configuration as a YAML file
4. Validate the configuration before training (optional):

```bash
python scripts/validate_config.py --config your_config.yaml
```

## Best Practices

- **Start small**: Begin with a smaller model or subset of data to verify your configuration
- **Validate parameters**: Ensure parameter combinations are compatible (e.g., QLoRA requires 4-bit quantization)
- **Monitor training**: Watch for signs of instability or poor convergence
- **Iterative refinement**: Adjust configuration based on initial results
- **Save configurations**: Keep track of configurations for reproducibility

## Troubleshooting

- **OOM errors**: Reduce batch size, use gradient checkpointing, or switch to QLoRA
- **Slow training**: Adjust optimizations like Flash Attention or mixed precision
- **Poor convergence**: Tune learning rate and scheduler
- **NaN losses**: Reduce learning rate, adjust gradient clipping
- **DeepSpeed errors**: Ensure compatible configuration with your hardware
