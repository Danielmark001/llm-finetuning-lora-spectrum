# Artemis – Adaptive Representation Tuning for Efficient Model Instruction Synthesis

A framework for parameter-efficient fine-tuning of large language models with significant performance and efficiency improvements.

## Core Features

- **Efficiency-Transformer**:

  - Parameter-efficient fine-tuning system with adaptive layer selection
  - Dynamic rank allocation based on layer importance
  - Gradient-based pruning during training
  - Cross-layer parameter sharing mechanisms

- **Advanced Pruning Techniques**:

  - Structured sparsity with automated threshold determination
  - Magnitude-based weight pruning with importance scoring
  - Progressive layer dropout during training
  - Quantization-aware pruning for compressed deployment

- **Hybrid LoRA-Adapter Approach**:

  - Combined benefits of LoRA and Adapter-based methods
  - 8-bit quantization with calibration for inference
  - Mixed-precision training with adaptive bit allocation
  - Hardware-aware optimization for consumer GPUs

- **Custom Evaluation Framework**:
  - Domain-specific benchmarks with automated task generation
  - Comparative analysis against full fine-tuning baselines
  - Resource utilization tracking and optimization metrics
  - Real-world application performance measurements

## Installation

```bash
# Clone the repository
git clone https://github.com/Danielmark001/llm-finetuning-lora-spectrum.git
cd llm-finetuning-lora-spectrum

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Prepare your dataset

Prepare your dataset in one of the supported formats (Alpaca, ShareGPT).

### 2. Configure training

Create or modify `config.yaml` with your desired training parameters:

```yaml
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
  load_in_8bit: true
  hybrid_lora_adapter: true
  pruning:
    enabled: true
    sparsity_target: 0.6
    method: "magnitude_progressive"

fine_tuning:
  method: "efficiency_transformer" # Options: "full", "lora", "qlora", "spectrum", "efficiency_transformer"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
      ]
    dynamic_rank: true
# Additional configuration...
```

### 3. Run training

```bash
python src/train.py --config config.yaml
```

For distributed training:

```bash
python scripts/distributed_setup.py launcher --config config.yaml --num_gpus_per_node 4 --use_deepspeed
bash scripts/run_training.sh
```

### 4. Evaluate the model

```bash
python scripts/inference.py --model_path path/to/model --adapter_path path/to/adapter --evaluate --eval_dataset path/to/eval_data.json --perplexity --benchmarks domain-specific
```

### 5. Run optimized inference

For interactive chat:

```bash
python scripts/inference.py --model_path path/to/model --adapter_path path/to/adapter --chat --load_in_8bit --hybrid_mode
```

For batch inference:

```bash
python scripts/inference.py --model_path path/to/model --adapter_path path/to/adapter --batch --input_file inputs.json --output_file outputs.json --load_in_8bit --hybrid_mode
```

### 6. Run demonstration

To see Artemis in action with a real-world example:

```bash
python scripts/demo.py --domain legal --model meta-llama/Llama-3.1-8B-Instruct
```

Available domains: `legal`, `medical`, and `customer-support`.

## Project Structure

```
Artemis/
├── config/               # Configuration files
├── data/                 # Sample datasets and data utilities
├── docs/                 # Documentation
│   ├── getting_started.md     # Quick start guide
│   ├── efficiency_transformer.md  # Efficiency-Transformer documentation
│   ├── pruning_techniques.md  # Pruning techniques documentation
│   ├── hybrid_lora_adapter.md # Hybrid LoRA-Adapter documentation
│   └── evaluation_framework.md # Evaluation framework documentation
├── evaluation/           # Evaluation datasets and benchmarks
├── models/               # Directory for saved models
├── notebooks/            # Tutorial notebooks
├── scripts/              # Utility scripts
│   ├── distributed_setup.py   # Distributed training setup
│   ├── inference.py      # Inference script for optimized models
│   ├── demo.py           # Demonstration script with real-world examples
│   └── ...
├── src/                  # Core implementation
│   ├── train.py          # Main training script with Artemis features
│   └── utils/            # Utility modules
│       ├── data_processing.py     # Dataset processing utilities
│       ├── efficiency.py          # Efficiency-Transformer implementation
│       ├── evaluation.py          # Evaluation utilities
│       ├── hybrid_adapter.py      # Hybrid LoRA-Adapter implementation
│       ├── optimization.py        # Optimization utilities
│       ├── pruning.py             # Pruning techniques
│       └── spectrum.py            # Spectrum implementation
└── README.md
```

## Key Commands

Here are some useful commands for working with Artemis:

```bash
# Create domain-specific benchmarks
python src/train.py --create_benchmarks

# Run evaluation only (no training)
python src/train.py --config config.yaml --eval_only

# Fine-tune with all Artemis optimizations
python src/train.py --config config.yaml

# Run optimized inference with 8-bit hybrid mode
python scripts/inference.py --model_path path/to/model --adapter_path path/to/adapter --chat --load_in_8bit --hybrid_mode

# Compare baseline vs. Artemis performance
python scripts/demo.py --domain medical
```

## Documentation

For detailed documentation, see the `docs/` directory:

- [Getting Started Guide](docs/getting_started.md)
- [Efficiency-Transformer Methods](docs/efficiency_transformer.md)
- [Pruning Techniques](docs/pruning_techniques.md)
- [Hybrid LoRA-Adapter Approach](docs/hybrid_lora_adapter.md)
- [Custom Evaluation Framework](docs/evaluation_framework.md)
- [Case Studies](docs/case_studies.md)


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
