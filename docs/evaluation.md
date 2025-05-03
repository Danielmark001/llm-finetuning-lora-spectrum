# Evaluation Guide for Fine-tuned LLMs

This document describes the comprehensive evaluation framework provided in this project and how to use it to assess your fine-tuned language models.

## Overview

Proper evaluation is essential for understanding the performance, capabilities, and limitations of your fine-tuned models. Our evaluation framework provides multiple methods to assess models from different perspectives:

1. **Perplexity Evaluation**: Measure the model's ability to predict text
2. **Benchmark Evaluation**: Test performance on standardized benchmarks
3. **Domain-Specific Evaluation**: Assess performance in specific domains
4. **Human Evaluation**: Tools for human assessment of model outputs
5. **Generation Quality**: Evaluate model-generated text with automated metrics

## Perplexity Evaluation

Perplexity is a statistical measure of how well a model predicts a sample of text. Lower perplexity indicates better performance.

### Usage

```python
from src.utils.evaluation import calculate_perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# Load evaluation dataset
dataset = load_dataset("json", data_files="data/eval.json")["train"]

# Calculate perplexity
results = calculate_perplexity(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    batch_size=4,
    max_length=2048,
    stride=512,
)

print(f"Perplexity: {results['perplexity']:.4f}")
```

### Interpreting Results

- Lower perplexity is better
- Compare against the base model to measure improvement
- Typical perplexity values vary by domain and model size
- Different domains may have different "good" perplexity ranges

## Benchmark Evaluation

The framework integrates with the LM Evaluation Harness to evaluate models on standardized benchmarks like MMLU, TruthfulQA, and HellaSWAG.

### Usage

```python
from src.utils.evaluation import evaluate_with_lm_harness

results = evaluate_with_lm_harness(
    model=model,
    tokenizer=tokenizer,
    tasks=["mmlu", "truthfulqa_mc", "hellaswag"],
    batch_size=8,
    output_path="evaluation/benchmark_results.json",
)

# Print summary of results
for task, metrics in results["results"].items():
    print(f"{task}: {metrics}")
```

### Available Benchmarks

- **MMLU**: Multi-task language understanding across 57 subjects
- **HellaSWAG**: Commonsense inference
- **TruthfulQA**: Measuring truthfulness and factuality
- **GSM8K**: Grade school math problems
- **ARC**: AI2 Reasoning Challenge
- **WinoGrande**: Winograd Schema Challenge
- And many more through the LM Evaluation Harness

## Domain-Specific Evaluation

For specialized use cases, domain-specific evaluation provides targeted assessment of the model's capabilities in specific fields.

### Supported Domains

- **Medical**: Medical knowledge and reasoning
- **Legal**: Legal understanding and reasoning
- **Code**: Programming and code generation
- **Reasoning**: Logical and mathematical reasoning

### Usage

```python
from src.utils.evaluation import run_domain_specific_evaluation

# Evaluate on medical domain
medical_results = run_domain_specific_evaluation(
    model=model,
    tokenizer=tokenizer,
    domain="medical",
    evaluation_data_path="evaluation/domain_medical.json",
    batch_size=4,
    output_path="evaluation/medical_results.json",
)

print(f"Medical evaluation results: {medical_results}")
```

### Creating Custom Domain Evaluations

You can create custom domain evaluations by:

1. Creating a JSON file with evaluation examples
2. Implementing a custom evaluation function
3. Registering it in the evaluation module

## Human Evaluation

For qualitative assessment, the framework provides tools to prepare model outputs for human evaluation.

### Usage

```python
from src.utils.evaluation import prepare_human_eval_interface, load_human_eval_results

# Generate outputs for human evaluation
outputs = generate_outputs(
    model=model,
    tokenizer=tokenizer,
    dataset=eval_dataset,
    prompt_column="instruction",
    max_new_tokens=512,
    batch_size=4,
)

# Prepare human evaluation interface
prepare_human_eval_interface(
    model_outputs=outputs,
    model_name="my-fine-tuned-model",
    output_path="evaluation/human_eval_data",
)

# After human evaluation is completed, load and analyze results
results = load_human_eval_results("evaluation/human_eval_data_completed.csv")
```

### Human Evaluation Criteria

When conducting human evaluations, consider the following criteria:

- **Relevance**: How relevant is the response to the prompt?
- **Accuracy**: Is the information factually correct?
- **Coherence**: Is the response well-structured and logical?
- **Helpfulness**: How helpful is the response in addressing the user's needs?
- **Safety**: Does the response adhere to safety guidelines?

## Comprehensive Evaluation

The `evaluate_model` function provides a comprehensive evaluation using multiple methods.

```python
from src.utils.evaluation import evaluate_model

results = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    benchmarks=["lm-evaluation-harness", "domain-specific-eval", "human-eval"],
    batch_size=8,
    output_dir="evaluation/results",
)
```

## Generation Quality Metrics

For evaluating the quality of generated text, the framework includes automatic metrics:

- **BLEU**: Measures n-gram overlap with reference text
- **ROUGE**: Recall-based overlap metrics
- **BERTScore**: Contextual similarity using BERT embeddings

```python
from src.utils.evaluation import calculate_text_metrics

metrics = calculate_text_metrics(
    references=reference_texts,
    predictions=generated_texts,
)

print(f"BLEU: {metrics['bleu']:.4f}")
print(f"ROUGE-L: {metrics['rougeL_f']:.4f}")
```

## Command-Line Evaluation

You can also run evaluation from the command line using the `inference.py` script:

```bash
python scripts/inference.py \
    --model_path path/to/model \
    --adapter_path path/to/adapter \
    --evaluate \
    --eval_dataset path/to/eval.json \
    --perplexity \
    --benchmarks lm-evaluation-harness domain-specific-eval \
    --eval_output_dir evaluation/results
```

## Evaluation Best Practices

1. **Compare against baselines**: Always evaluate the base model as a baseline
2. **Use multiple metrics**: Don't rely on a single metric for assessment
3. **Include human evaluation**: Automated metrics don't capture all aspects of quality
4. **Test edge cases**: Specifically evaluate challenging or edge cases
5. **Domain relevance**: Use domain-specific evaluation for specialized models
6. **Iterative improvement**: Use evaluation results to guide further fine-tuning
7. **Evaluate regularly**: Track performance throughout the training process

## Common Pitfalls

- **Overfitting**: High performance on training data but poor generalization
- **Metric gaming**: Optimizing for metrics at the expense of actual quality
- **Benchmark leakage**: Using benchmark data in training
- **Selection bias**: Evaluating only on favorable examples
- **Ignoring failure modes**: Not examining where the model performs poorly

## Advanced Evaluation Techniques

### Robustness Testing

Test the model's robustness to input variations:

```python
def evaluate_robustness(model, tokenizer, prompts):
    variations = []
    for prompt in prompts:
        variations.append(prompt)  # Original
        variations.append(prompt.lower())  # Lowercase
        variations.append(" ".join(prompt.split()))  # Extra spaces
        # Add typos, etc.
    
    results = []
    for var in variations:
        # Generate and score outputs
        # ...
    
    return robustness_score
```

### Comparative Evaluation

Compare multiple models side-by-side:

```python
def comparative_evaluation(models, tokenizer, prompts):
    results = {}
    for model_name, model in models.items():
        results[model_name] = generate_and_evaluate(model, tokenizer, prompts)
    
    return results
```

### Challenge Sets

Create specialized challenge sets to test specific capabilities:

```python
challenge_sets = {
    "logical_reasoning": [...],
    "factual_recall": [...],
    "math_problems": [...],
    # ...
}

for challenge, prompts in challenge_sets.items():
    results = evaluate_on_prompts(model, tokenizer, prompts)
    print(f"{challenge}: {results}")
```

## Conclusion

A comprehensive evaluation strategy is essential for understanding your fine-tuned model's capabilities and limitations. Use multiple evaluation approaches, including both automatic metrics and human assessment, to get a complete picture of model performance. The evaluation results should guide further iterations of fine-tuning and help you make informed decisions about model deployment.
