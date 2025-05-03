"""
Data Processing Utilities for LLM Fine-Tuning
=============================================
This module contains functions for preprocessing and handling datasets
for LLM fine-tuning, including specialized formats like Alpaca, ShareGPT, etc.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def format_alpaca_prompt(
    row: Dict[str, Any],
    add_system_prompt: bool = True,
    system_prompt: str = "You are a helpful assistant that responds accurately, helpfully, and harmlessly.",
) -> str:
    """
    Format a prompt in Alpaca format.
    
    Args:
        row: Dataset row containing 'instruction', 'input', etc.
        add_system_prompt: Whether to add a system prompt
        system_prompt: System prompt text to use
        
    Returns:
        str: Formatted prompt
    """
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    
    # Create prompt based on whether input is provided
    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
    else:
        prompt = f"Instruction: {instruction}\nOutput: "
    
    # Add system prompt if requested
    if add_system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    
    return prompt


def format_sharegpt_prompt(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> str:
    """
    Format a prompt in ShareGPT format.
    
    Args:
        row: Dataset row containing 'conversations' list
        tokenizer: Tokenizer with chat template
        
    Returns:
        str: Formatted prompt using the tokenizer's chat template
    """
    conversations = row.get("conversations", [])
    if not conversations:
        return ""
    
    # Convert to the format expected by the chat template
    messages = []
    for msg in conversations:
        role = msg.get("role", "").lower()
        
        # Map roles to standard format
        if role == "human":
            role = "user"
        elif role in ["gpt", "assistant", "bot"]:
            role = "assistant"
        
        messages.append({"role": role, "content": msg.get("value", "")})
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return prompt


def process_alpaca_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    add_eos_token: bool = True,
    add_bos_token: bool = False,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
) -> Dataset:
    """
    Process a dataset in Alpaca format for fine-tuning.
    
    Args:
        dataset: Input dataset with instruction, input, and output fields
        tokenizer: Tokenizer for the target model
        max_seq_length: Maximum sequence length
        add_eos_token: Whether to add EOS token at the end
        add_bos_token: Whether to add BOS token at the beginning
        use_chat_template: Whether to use the tokenizer's chat template
        system_prompt: Optional system prompt to include
        
    Returns:
        Dataset: Processed dataset ready for training
    """
    def _process_row(row):
        # Extract fields
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        output = row.get("output", "")
        
        # Use chat template if available and requested
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Format query based on whether input is provided
            if input_text:
                content = f"{instruction}\n\n{input_text}"
            else:
                content = instruction
            
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": output})
            
            # Apply the model's chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Format as a simple prompt-response pair
            if input_text:
                text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            else:
                text = f"Instruction: {instruction}\nOutput: {output}"
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        # Add BOS/EOS tokens if needed
        input_ids = tokenized["input_ids"]
        if add_bos_token and input_ids[0] != tokenizer.bos_token_id:
            input_ids = [tokenizer.bos_token_id] + input_ids
        if add_eos_token and input_ids[-1] != tokenizer.eos_token_id:
            input_ids = input_ids + [tokenizer.eos_token_id]
        
        # Update tokenized output
        tokenized["input_ids"] = input_ids
        tokenized["attention_mask"] = [1] * len(input_ids)
        
        # Add labels for autoregressive training
        tokenized["labels"] = input_ids.copy()
        
        return tokenized
    
    # Apply processing to each row
    return dataset.map(_process_row)


def process_sharegpt_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    add_eos_token: bool = True,
) -> Dataset:
    """
    Process a dataset in ShareGPT format for fine-tuning.
    
    Args:
        dataset: Input dataset with conversations field
        tokenizer: Tokenizer for the target model
        max_seq_length: Maximum sequence length
        add_eos_token: Whether to add EOS token at the end
        
    Returns:
        Dataset: Processed dataset ready for training
    """
    def _process_row(row):
        conversations = row.get("conversations", [])
        if not conversations:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # Convert conversations to the format expected by the chat template
        messages = []
        for msg in conversations:
            role = msg.get("role", "").lower()
            
            # Map roles to standard format
            if role == "human":
                role = "user"
            elif role in ["gpt", "assistant", "bot"]:
                role = "assistant"
            
            messages.append({"role": role, "content": msg.get("value", "")})
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        # Add EOS token if needed
        input_ids = tokenized["input_ids"]
        if add_eos_token and input_ids[-1] != tokenizer.eos_token_id:
            input_ids = input_ids + [tokenizer.eos_token_id]
        
        # Update tokenized output
        tokenized["input_ids"] = input_ids
        tokenized["attention_mask"] = [1] * len(input_ids)
        
        # Add labels for autoregressive training
        tokenized["labels"] = input_ids.copy()
        
        return tokenized
    
    # Apply processing to each row
    return dataset.map(_process_row)


def preprocess_dataset(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    tokenizer: PreTrainedTokenizer = None,
    max_seq_length: int = 4096,
    format: str = "alpaca",
    add_eos_token: bool = True,
    add_bos_token: bool = False,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
) -> Dict[str, Dataset]:
    """
    Preprocess datasets for fine-tuning based on their format.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        tokenizer: Tokenizer for the target model
        max_seq_length: Maximum sequence length
        format: Dataset format (alpaca, sharegpt, oasst, custom)
        add_eos_token: Whether to add EOS token
        add_bos_token: Whether to add BOS token
        use_chat_template: Whether to use the tokenizer's chat template
        system_prompt: Optional system prompt
        
    Returns:
        Dict[str, Dataset]: Processed datasets
    """
    result = {}
    
    if format == "alpaca":
        logger.info("Processing dataset in Alpaca format")
        result["train"] = process_alpaca_dataset(
            train_dataset,
            tokenizer,
            max_seq_length,
            add_eos_token,
            add_bos_token,
            use_chat_template,
            system_prompt,
        )
        
        if eval_dataset:
            result["eval"] = process_alpaca_dataset(
                eval_dataset,
                tokenizer,
                max_seq_length,
                add_eos_token,
                add_bos_token,
                use_chat_template,
                system_prompt,
            )
    
    elif format == "sharegpt":
        logger.info("Processing dataset in ShareGPT format")
        result["train"] = process_sharegpt_dataset(
            train_dataset,
            tokenizer,
            max_seq_length,
            add_eos_token,
        )
        
        if eval_dataset:
            result["eval"] = process_sharegpt_dataset(
                eval_dataset,
                tokenizer,
                max_seq_length,
                add_eos_token,
            )
    
    # Add support for other formats as needed
    else:
        raise ValueError(f"Unsupported dataset format: {format}")
    
    return result


def calculate_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Calculate statistics about a tokenized dataset.
    
    Args:
        dataset: Tokenized dataset
        
    Returns:
        Dict[str, Any]: Statistics about the dataset
    """
    # Get sequence lengths
    seq_lengths = [len(x["input_ids"]) for x in dataset]
    
    # Calculate statistics
    stats = {
        "count": len(dataset),
        "min_length": min(seq_lengths),
        "max_length": max(seq_lengths),
        "mean_length": sum(seq_lengths) / len(seq_lengths),
        "median_length": sorted(seq_lengths)[len(seq_lengths) // 2],
        "total_tokens": sum(seq_lengths),
    }
    
    return stats
