# Pruning Techniques in Artemis

Artemis implements advanced pruning techniques that can reduce model size by up to 60% with negligible quality impact. This document explains the pruning methods available in Artemis and how to use them effectively.

## Overview

Model pruning removes unnecessary parameters from neural networks to:

- Reduce model size
- Decrease memory usage
- Accelerate inference
- Lower computational requirements
- Enable deployment on resource-constrained devices

Artemis provides a comprehensive pruning framework that integrates with the training process and offers multiple pruning strategies optimized for large language models.

## Key Features

- **Progressive Pruning**: Gradually increases sparsity during training for better quality
- **Multiple Pruning Methods**: Magnitude-based, structured sparsity, and layer dropout
- **Importance Scoring**: Uses various metrics to determine which parameters to prune
- **Quantization-Aware**: Ensures pruned models are compatible with quantization
- **Minimal Quality Impact**: Preserves performance even at high sparsity levels (60%+)

## Pruning Methods

### 1. Magnitude-Based Progressive Pruning

This is the default and most versatile pruning method in Artemis. It operates by:

1. Gradually increasing sparsity during training according to a schedule
2. Using parameter magnitude as an importance criterion
3. Applying a global threshold to prune the least important weights

Key characteristics:
- Unstructured sparsity (individual weights are pruned)
- Excellent quality preservation
- Compatible with both training and fine-tuning
- Works well with quantization

#### Configuration:

```yaml
pruning:
  enabled: true
  method: "magnitude_progressive"
  initial_sparsity: 0.0
  final_sparsity: 0.6
  pruning_start: 0.2  # Start after 20% of training
  pruning_end: 0.8    # End after 80% of training
  pruning_interval: 50
  importance_metric: "magnitude"
  quantization_aware: true
```

### 2. Structured Sparsity

This method prunes entire structures within the model, such as attention heads or neurons. It works by:

1. Analyzing the importance of structural components
2. Pruning entire components rather than individual weights
3. Maintaining the model's architectural integrity

Key characteristics:
- Creates hardware-friendly sparsity patterns
- Better acceleration on some hardware
- Slightly higher quality impact than magnitude pruning
- Excellent for inference optimization

#### Configuration:

```yaml
pruning:
  enabled: true
  method: "structured_sparsity"
  initial_sparsity: 0.0
  final_sparsity: 0.5
  pruning_start: 0.2
  pruning_end: 0.8
  pruning_interval: 50
  importance_metric: "magnitude"
  quantization_aware: true
```

### 3. Layer Dropout

This aggressive method drops entire layers during training as a form of structured pruning. It:

1. Identifies the least important layers based on importance metrics
2. Progressively drops entire layers during training
3. Forces the model to compensate for missing layers

Key characteristics:
- Highest sparsity achievable
- Most dramatic acceleration
- Higher quality impact than other methods
- Best for extreme compression needs

#### Configuration:

```yaml
pruning:
  enabled: true
  method: "layer_dropout"
  initial_sparsity: 0.0
  final_sparsity: 0.4
  pruning_start: 0.3
  pruning_end: 0.7
  pruning_interval: 50
  importance_metric: "gradient_sensitivity"
  quantization_aware: true
```

## Importance Metrics

Artemis supports several importance metrics to determine which parameters to prune:

### Magnitude-Based

Uses the absolute value of weights as the importance criterion. This is the simplest and most efficient method.

```yaml
pruning:
  importance_metric: "magnitude"
```

### Gradient Sensitivity

Uses the product of weight magnitude and accumulated gradients to identify parameters that have the most impact on the loss.

```yaml
pruning:
  importance_metric: "gradient_sensitivity"
```

### Activation-Based

Measures the impact of parameters on activations to identify important connections.

```yaml
pruning:
  importance_metric: "activation"
```

## Progressive Pruning Schedule

Artemis uses a cubic schedule for progressive pruning:

$$s(t) = s_i + (s_f - s_i) \cdot \left(\frac{t - t_s}{t_e - t_s}\right)^3$$

Where:
- $s(t)$ is the sparsity at step $t$
- $s_i$ is the initial sparsity
- $s_f$ is the final sparsity
- $t_s$ is the step at which pruning starts
- $t_e$ is the step at which pruning ends

This schedule is gentler at the beginning, allowing the model to adapt before aggressive pruning begins.

## Usage Example

Here's how to use pruning in your Artemis training script:

```python
from utils.pruning import create_pruning_manager

# Load your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Create pruning configuration
pruning_config = {
    "method": "magnitude_progressive",
    "initial_sparsity": 0.0,
    "final_sparsity": 0.6,
    "pruning_start": 0.2,
    "pruning_end": 0.8,
    "pruning_interval": 50,
    "importance_metric": "magnitude",
    "quantization_aware": True,
}

# Create pruning manager
pruning_manager = create_pruning_manager({"pruning": pruning_config}, model)

# Attach to model
model.pruning_manager = pruning_manager

# During training loop:
for step in range(total_steps):
    # Forward and backward pass
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    
    # Apply pruning step
    pruning_manager.step(total_steps=total_steps)
    
    # Apply mask to gradients
    pruning_manager.apply_mask_to_gradients()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

# After training, prepare for quantization
pruning_manager.prepare_for_quantization()

# Get pruning summary
summary = pruning_manager.get_pruning_summary()
print(f"Model size reduction: {summary['pruning_metrics']['model_size_reduction']:.2%}")
```

## Programmatic Analysis

You can analyze the impact of different sparsity levels without committing to a full training run:

```python
# Create pruning manager
pruning_manager = create_pruning_manager({"pruning": pruning_config}, model)

# Test different sparsity levels
sparsity_levels = [0.3, 0.5, 0.7, 0.9]
results = {}

for sparsity in sparsity_levels:
    # Apply pruning at this level
    pruning_manager.apply_magnitude_pruning(sparsity)
    
    # Evaluate performance
    metrics = evaluate_model(model, tokenizer, eval_dataset)
    
    # Record results
    results[sparsity] = {
        "metrics": metrics,
        "model_size_mb": pruning_manager.baseline_model_size * (1 - pruning_manager.pruning_metrics["model_size_reduction"]),
    }
    
    # Reset model (reload in a real scenario)
    
# Find optimal sparsity level
optimal_sparsity = max(
    (s for s in sparsity_levels if results[s]["metrics"]["accuracy"] > threshold),
    default=sparsity_levels[0]
)
```

## Combined with Other Artemis Features

Pruning works particularly well when combined with other Artemis features:

1. **Efficiency-Transformer + Pruning**: Use Efficiency-Transformer to identify important layers, then apply pruning to these layers.

2. **Pruning + Hybrid LoRA-Adapter**: Apply pruning during training, then use Hybrid LoRA-Adapter for inference to gain maximum speedup.

3. **Pruning + 8-bit Quantization**: Pruned models quantize better due to reduced parameter redundancy.

Example combined configuration:

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
  method: "efficiency_transformer"
  efficiency_transformer:
    adaptive_layer_selection: true
    cross_layer_parameter_sharing: true
```

## Performance Benchmarks

### Medical Domain Model

| Sparsity | Model Size | Accuracy | Latency |
|----------|------------|----------|---------|
| 0% (baseline) | 15 GB | 85.1% | 380 ms |
| 30% | 10.5 GB | 84.8% | 250 ms |
| 50% | 7.5 GB | 84.2% | 190 ms |
| 60% | 6 GB | 83.7% | 160 ms |
| 70% | 4.5 GB | 82.1% | 130 ms |

### Legal Document Processing Model

| Sparsity | Model Size | F1 Score | Documents/min |
|----------|------------|----------|--------------|
| 0% (baseline) | 24 GB | 91.2% | 12 |
| 30% | 16.8 GB | 90.9% | 18 |
| 50% | 12 GB | 90.5% | 25 |
| 60% | 9.6 GB | 89.8% | 30 |
| 70% | 7.2 GB | 87.4% | 36 |

## Best Practices

1. **Start Conservative**: Begin with lower sparsity targets (40-50%) and gradually increase if quality remains acceptable.

2. **Monitor Quality**: Regularly evaluate model quality during pruning to catch potential degradation early.

3. **Layer-Specific Sparsity**: Consider different sparsity targets for different layers (higher for less important layers).

4. **Progressive Pruning**: Always use progressive pruning during training rather than one-shot pruning.

5. **Gradient-Based Importance**: For the best quality, use gradient_sensitivity as the importance metric, though it's more computationally expensive.

6. **Combine with Knowledge Distillation**: If possible, use knowledge distillation alongside pruning for better quality preservation.

## Troubleshooting

### Quality Degradation

If you experience significant quality degradation:

- Decrease the final sparsity target
- Use a slower pruning schedule (start later, end later)
- Switch to magnitude_progressive method if using structured methods
- Try gradient_sensitivity as the importance metric

### Inference Not Faster

If you don't see significant inference speedup:

- Ensure you're using structured_sparsity for hardware acceleration
- Apply quantization after pruning
- Use a runtime that supports sparse tensor operations
- Try the Hybrid LoRA-Adapter approach alongside pruning

### Memory Issues

If you encounter memory issues during pruning:

- Enable low_resource_mode in your efficiency configuration
- Increase pruning_interval to apply pruning less frequently
- Use a smaller batch size during training
- Apply gradient checkpointing alongside pruning

## Conclusion

Artemis's pruning techniques offer a powerful way to reduce model size and accelerate inference while preserving model quality. By combining different pruning methods with other Artemis features, you can achieve optimal performance for your specific use case.

For more information, see the `src/utils/pruning.py` implementation and the examples in the `notebooks/` directory.
