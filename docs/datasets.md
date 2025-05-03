# Dataset Preparation Guide

This guide explains how to prepare and format datasets for LLM fine-tuning using our framework.

## Supported Formats

The framework supports the following dataset formats:

1. **Alpaca Format** - Instruction-based dataset format with instruction, input (optional), and output fields
2. **ShareGPT Format** - Conversational dataset format with a series of exchanges between users and assistants
3. **Custom Formats** - Extensible support for custom dataset formats

## Alpaca Format

The Alpaca format is a simple instruction-based format that consists of:

- **instruction**: The main instruction or question
- **input**: Optional additional context or input information
- **output**: The target response the model should generate

### Example

```json
[
  {
    "instruction": "Write a poem about machine learning.",
    "input": "",
    "output": "Silicon thoughts in matrices align,\nPatterns emerge where chaos once defined.\nThrough layers deep the data streams,\nLearning more than human dreams.\n\nWith every epoch knowledge grows,\nA digital mind that ebbs and flows.\nNeither alive nor truly still,\nA testament to human will."
  },
  {
    "instruction": "Translate the following sentence to French.",
    "input": "I love artificial intelligence and machine learning.",
    "output": "J'aime l'intelligence artificielle et l'apprentissage automatique."
  }
]
```

### Processing

The framework processes Alpaca format datasets by:

1. Combining instruction, input, and output into a single text sequence
2. Applying tokenization and optional chat formatting
3. Creating appropriate labels for training

## ShareGPT Format

The ShareGPT format is designed for conversational data with multiple turns between users and assistants.

### Example

```json
[
  {
    "conversations": [
      {
        "role": "human",
        "value": "Hello, how are you today?"
      },
      {
        "role": "assistant",
        "value": "I'm doing well, thank you for asking! How can I help you today?"
      },
      {
        "role": "human",
        "value": "Can you explain what machine learning is?"
      },
      {
        "role": "assistant",
        "value": "Machine learning is a branch of artificial intelligence that enables computers to learn and make predictions from data without being explicitly programmed. Instead of following predefined rules, ML algorithms identify patterns in data and improve their performance through experience."
      }
    ]
  }
]
```

### Processing

ShareGPT datasets are processed by:

1. Converting the conversation format to the model's expected chat template
2. Applying tokenization with appropriate masks for human/assistant roles
3. Creating labels for autoregressive training

## Dataset Preparation Steps

### 1. Data Collection

Collect high-quality examples that represent the tasks you want the model to perform. Consider:

- **Diversity**: Include a wide range of instructions/conversations
- **Quality**: Ensure responses are high-quality, factual, and helpful
- **Balance**: Balance different types of tasks in your dataset

### 2. Data Cleaning

Clean your data to remove:

- Duplicates
- Inappropriate content
- Low-quality examples
- Personal identifiable information (PII)
- Formatting issues

### 3. Format Conversion

Convert your data to one of the supported formats (Alpaca or ShareGPT).

### 4. Dataset Splitting

Split your dataset into training and evaluation sets:

```python
from sklearn.model_selection import train_test_split

with open('your_dataset.json', 'r') as f:
    data = json.load(f)
    
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)

with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=2)
    
with open('eval.json', 'w') as f:
    json.dump(eval_data, f, indent=2)
```

### 5. Configuration

Update your configuration file to point to your dataset:

```yaml
dataset:
  format: "alpaca"  # or "sharegpt"
  train_path: "data/train.json"
  eval_path: "data/eval.json"
  preprocessing:
    add_eos_token: true
    add_bos_token: false
    use_chat_template: true
```

## Advanced Processing

### Chat Templates

Most modern LLMs support chat formatting via templates. The framework can automatically apply the appropriate template based on the model you're using.

For example, Llama models typically use a format like:

```
<|system|>
You are a helpful assistant.
</|system|>

<|user|>
How does fine-tuning work?
</|user|>

<|assistant|>
Fine-tuning is a technique...
</|assistant|>
```

While other models might use different formats. The tokenizer's `apply_chat_template` method handles these differences automatically.

### System Prompts

You can include system prompts in your dataset to guide the model's behavior:

```json
[
  {
    "conversations": [
      {
        "role": "system",
        "value": "You are a helpful, harmless, and honest assistant."
      },
      {
        "role": "human",
        "value": "What is the capital of France?"
      },
      {
        "role": "assistant",
        "value": "The capital of France is Paris."
      }
    ]
  }
]
```

### Token Length Considerations

Be mindful of the token length of your examples:

- Ensure examples fit within the model's context window
- For Llama-3.1 models, the context window is typically 8192 tokens
- Balance between short and longer examples in your dataset

### Using the DataProcessor

The framework includes data processing utilities to handle these formats:

```python
from src.utils.data_processing import preprocess_dataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load dataset
dataset = load_dataset("json", data_files="data/your_dataset.json")["train"]

# Process dataset
processed_dataset = preprocess_dataset(
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=4096,
    format="alpaca",  # or "sharegpt"
    add_eos_token=True,
    use_chat_template=True
)
```

## Sample Datasets

The `data/` directory contains sample datasets for demonstration:

- `sample_alpaca.json`: A small example of the Alpaca format
- `sample_sharegpt.json`: A small example of the ShareGPT format

## Extending to Custom Formats

To support custom dataset formats, you can extend the `preprocess_dataset` function in `src/utils/data_processing.py`:

1. Create a processing function for your format
2. Add a new condition in the `preprocess_dataset` function
3. Update the documentation to include your custom format

Example implementation for a custom format:

```python
def process_custom_format(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    add_eos_token: bool = True,
) -> Dataset:
    """
    Process a dataset in your custom format.
    
    Args:
        dataset: Input dataset with your custom fields
        tokenizer: Tokenizer for the target model
        max_seq_length: Maximum sequence length
        add_eos_token: Whether to add EOS token
        
    Returns:
        Dataset: Processed dataset ready for training
    """
    # Your custom processing logic here
    ...
    
    return processed_dataset
```

Then in `preprocess_dataset`:

```python
elif format == "custom":
    logger.info("Processing dataset in custom format")
    result["train"] = process_custom_format(
        train_dataset,
        tokenizer,
        max_seq_length,
        add_eos_token,
    )
    
    if eval_dataset:
        result["eval"] = process_custom_format(
            eval_dataset,
            tokenizer,
            max_seq_length,
            add_eos_token,
        )
```

## Best Practices

1. **Quality over quantity**: A smaller, high-quality dataset often produces better results than a large, noisy one.

2. **Domain focus**: For domain-specific fine-tuning, focus on examples that represent the target domain well.

3. **Balanced examples**: Ensure a good balance of different instruction types and response lengths.

4. **Validate your dataset**: Check a sample of processed examples to ensure they are formatted correctly before starting a long training run.

5. **Calculate statistics**: Use the `calculate_dataset_statistics` function to analyze token length distribution and other metrics.

```python
from src.utils.data_processing import calculate_dataset_statistics

stats = calculate_dataset_statistics(processed_dataset["train"])
print(f"Average token length: {stats['mean_length']}")
print(f"Maximum token length: {stats['max_length']}")
print(f"Total tokens: {stats['total_tokens']:,}")
```
