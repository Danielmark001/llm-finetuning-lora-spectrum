# LLM Fine-Tuning Framework

A comprehensive framework for efficient fine-tuning of large language models using state-of-the-art techniques.

## Features

- **Multiple Fine-Tuning Methods**:
  - Full parameter fine-tuning
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized Low-Rank Adaptation)
  - Spectrum (Signal-to-Noise Ratio based layer selection)

- **Efficient Processing**:
  - Support for multiple dataset formats (Alpaca, ShareGPT)
  - Chat template support
  - Optimized tokenization and batching

- **Distributed Training**:
  - DeepSpeed integration
  - FSDP (Fully Sharded Data Parallel) support
  - Multi-node configuration

- **Evaluation**:
  - Perplexity calculation
  - Integration with LM Evaluation Harness
  - Domain-specific evaluation
  - Human evaluation tools

- **Inference and Deployment**:
  - Interactive chat interface
  - Batch inference
  - Model deployment utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/username/llm-finetuning-project.git
cd llm-finetuning-project

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Prepare your dataset

Prepare your dataset in one of the supported formats (Alpaca, ShareGPT). Example of Alpaca format:

```json
[
  {
    "instruction": "Tell me a joke.",
    "input": "",
    "output": "Why don't scientists trust atoms? Because they make up everything!"
  }
]
```

### 2. Configure training

Create or modify `config.yaml` with your desired training parameters:

```yaml
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
  load_in_4bit: true
  use_flash_attention: true

fine_tuning:
  method: "qlora"  # Options: "full", "lora", "qlora", "spectrum"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
python scripts/inference.py --model_path path/to/model --adapter_path path/to/adapter --evaluate --eval_dataset path/to/eval_data.json --perplexity --benchmarks lm-evaluation-harness
```

### 5. Run inference

Interactive chat:

```bash
python scripts/inference.py --model_path path/to/model --adapter_path path/to/adapter --chat --load_in_4bit
```

Batch inference:

```bash
python scripts/inference.py --model_path path/to/model --adapter_path path/to/adapter --batch --input_file inputs.json --output_file outputs.json --load_in_4bit
```

## Project Structure

```
llm-finetuning-project/
├── config/               # Configuration files
├── data/                 # Sample datasets and data utilities
├── docs/                 # Documentation
├── evaluation/           # Evaluation datasets and benchmarks
├── models/               # Directory for saved models
├── notebooks/            # Tutorial notebooks
├── scripts/              # Utility scripts
│   ├── distributed_setup.py  # Distributed training setup
│   ├── inference.py      # Inference script
│   └── ...
├── src/                  # Core implementation
│   ├── train.py          # Main training script
│   └── utils/            # Utility modules
│       ├── data_processing.py  # Dataset processing utilities
│       ├── evaluation.py       # Evaluation utilities
│       ├── optimization.py     # Optimization utilities
│       └── spectrum.py         # Spectrum implementation
└── README.md
```

## Documentation

For detailed documentation, see the `docs/` directory:

- [Installation Guide](docs/installation.md)
- [Fine-Tuning Methods](docs/fine_tuning_methods.md)
- [Configuration Options](docs/configuration.md)
- [Dataset Preparation](docs/datasets.md)
- [Evaluation Guide](docs/evaluation.md)
- [Distributed Training](docs/distributed_training.md)
- [Inference and Deployment](docs/inference.md)

## Tutorials

Interactive tutorials are available in the `notebooks/` directory:

- [Fine-Tuning Tutorial](notebooks/fine_tuning_tutorial.ipynb)
- [Data Preparation](notebooks/data_preparation.ipynb)
- [Evaluation and Analysis](notebooks/evaluation.ipynb)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Spectrum method is based on the research paper: "Spectrum: Analyzing and Exploiting the Language Model's Frequency Domain for Parameter-Efficient Fine-Tuning"
- LoRA implementation is based on the [PEFT](https://github.com/huggingface/peft) library from Hugging Face
- Parts of the evaluation framework are inspired by the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
