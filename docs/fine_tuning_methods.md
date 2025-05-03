# Fine-Tuning Methods

This document explains the different fine-tuning methods available in the framework and provides guidance on when to use each method.

## Available Methods

The framework supports the following fine-tuning methods:

1. **Full Parameter Fine-tuning**
2. **LoRA (Low-Rank Adaptation)**
3. **QLoRA (Quantized Low-Rank Adaptation)**
4. **Spectrum (Signal-to-Noise Ratio based layer selection)**

## Method Comparison

| Method | Parameters Updated | Memory Usage | Training Speed | Performance | Use Case |
|--------|-------------------|--------------|----------------|-------------|----------|
| Full | All | Very High | Slow | Excellent | When you have significant compute resources |
| LoRA | Subset (rank adaptation) | Medium | Fast | Very Good | Balance between efficiency and performance |
| QLoRA | Subset (quantized) | Low | Medium | Good | Limited GPU memory |
| Spectrum | Selective layers | Medium-Low | Medium-Fast | Good | When you want to target specific model capabilities |

## Full Parameter Fine-tuning

Full parameter fine-tuning updates all parameters in the model. This is the most comprehensive approach but requires significant computational resources.

### Configuration

```yaml
fine_tuning:
  method: "full"
```

### When to use

- You have access to substantial GPU resources (multiple high-memory GPUs)
- You want the best possible performance
- You're training on a large, high-quality dataset
- You're fine-tuning a relatively smaller model (7B parameters or less)

## LoRA (Low-Rank Adaptation)

LoRA adds low-rank adaptation matrices to specific layers of the model, dramatically reducing the number of trainable parameters.

### Configuration

```yaml
fine_tuning:
  method: "lora"
  lora:
    r: 16  # Rank of the adaptation matrices
    alpha: 32  # Scaling factor
    dropout: 0.05  # Dropout rate
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias: "none"  # Whether to train bias parameters
```

### Key Parameters

- **r**: Rank of the adaptation matrices. Lower values mean fewer parameters to train but may reduce capacity.
- **alpha**: Scaling factor for the LoRA adaptations. Higher values give more weight to the adaptations.
- **target_modules**: Which module types to apply LoRA to. Common choices include attention layers and MLP layers.

### When to use

- Limited GPU memory (can work with consumer GPUs like RTX 3090/4090)
- You want efficient fine-tuning with good performance
- You plan to switch between multiple fine-tuned versions of the same base model

## QLoRA (Quantized Low-Rank Adaptation)

QLoRA extends LoRA by keeping the base model in 4-bit quantized format, reducing memory requirements even further.

### Configuration

```yaml
fine_tuning:
  method: "qlora"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias: "none"
  quantization:
    bits: 4
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
```

### When to use

- Very limited GPU memory (can work with 16GB or even 12GB GPUs)
- You're fine-tuning a larger model (13B+ parameters)
- You're willing to accept a slight performance drop for much better memory efficiency

## Spectrum (Signal-to-Noise Ratio based layer selection)

Spectrum analyzes the model's layers and selectively fine-tunes those with high signal-to-noise ratios, optimizing for efficiency and effectiveness.

### Configuration

```yaml
fine_tuning:
  method: "spectrum"
  spectrum:
    snr_threshold: 0.5  # SNR percentile threshold (0.0 to 1.0)
    layers_to_finetune: "auto"  # "auto" or a list of layer indices
```

### Key Parameters

- **snr_threshold**: Percentile threshold for selecting layers. Higher values mean fewer layers will be selected.
- **layers_to_finetune**: Set to "auto" for automatic selection based on SNR analysis, or specify a list of layer indices manually.

### When to use

- You want to target specific capabilities of the model
- You want to understand which layers are most important for your tasks
- You have limited compute and need a principled way to select which parts of the model to fine-tune

## Implementation Details

### LoRA and QLoRA

These methods are implemented using the PEFT (Parameter-Efficient Fine-Tuning) library from Hugging Face. The implementation adds low-rank matrices to the specified target modules.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
    bias=config["lora"]["bias"],
    task_type=TaskType.CAUSAL_LM,
    target_modules=config["lora"]["target_modules"],
)
model = get_peft_model(model, lora_config)
```

### Spectrum

Spectrum analyzes each layer's weights to calculate a signal-to-noise ratio (SNR), which is used to determine which layers are most important to fine-tune.

```python
analyzer = SpectrumAnalyzer(model)
snr_threshold = config["spectrum"]["snr_threshold"]
trainable_layers = analyzer.get_trainable_layers_by_snr(snr_threshold)
analyzer.freeze_layers_except(trainable_layers)
```

## Recommendations

1. **Start with QLoRA** if you have limited GPU resources. It provides a good balance of efficiency and performance.

2. **Try Spectrum** if you want to target specific capabilities or need an even more parameter-efficient approach.

3. **Use full fine-tuning** only if you have access to significant compute resources and want the best possible performance.

4. **Experiment with different LoRA ranks (r)** to find the right balance between parameter efficiency and model capacity for your task.

5. **Consider mixed approaches** such as using Spectrum to identify important layers, then applying LoRA only to those layers.

## Advanced Configuration

For advanced users, you can customize which modules are targeted by LoRA. Different models may have different naming conventions:

- **Llama/Mistral/Gemma**: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
- **GPT models**: "c_attn", "c_proj", "c_fc"

The framework automatically detects the model architecture and applies appropriate defaults, but you can override these in the configuration.
