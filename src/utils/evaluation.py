"""
Evaluation Utilities for LLM Fine-Tuning
========================================
This module provides comprehensive evaluation tools for fine-tuned language models,
including perplexity calculation, integration with evaluation harnesses,
and human evaluation interfaces.
"""

import os
import json
import logging
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TextGenerationPipeline,
    GenerationConfig,
)
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 8,
    max_length: int = 4096,
    stride: int = 512,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        dataset: Dataset containing text samples
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        stride: Stride for sliding window
        device: Device to use (defaults to model's device)
        
    Returns:
        Dict[str, float]: Perplexity metrics
    """
    logger.info("Calculating perplexity on dataset...")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Set device
    if device is None:
        device = model.device
    else:
        device = torch.device(device)
    
    # Set up metrics storage
    total_loss = 0
    total_tokens = 0
    
    # Function to get text samples from dataset
    def get_text_samples(dataset):
        if "text" in dataset.column_names:
            return dataset["text"]
        elif "content" in dataset.column_names:
            return dataset["content"]
        elif "output" in dataset.column_names:
            # For instruction datasets, concatenate instruction, input, and output
            outputs = []
            for sample in dataset:
                text = sample.get("output", "")
                if "instruction" in sample:
                    prefix = sample["instruction"]
                    if "input" in sample and sample["input"]:
                        prefix += "\n" + sample["input"]
                    text = prefix + "\n" + text
                outputs.append(text)
            return outputs
        elif "conversations" in dataset.column_names:
            # For conversation datasets
            outputs = []
            for sample in dataset:
                text = ""
                for turn in sample["conversations"]:
                    role = turn.get("role", "")
                    content = turn.get("value", "")
                    text += f"{role}: {content}\n"
                outputs.append(text)
            return outputs
        else:
            raise ValueError(
                "Dataset format not recognized. Must contain 'text', 'content', 'output', or 'conversations' column."
            )
    
    # Get text samples
    texts = get_text_samples(dataset)
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch_texts = texts[i:i+batch_size]
        batch_loss = 0
        batch_tokens = 0
        
        for text in batch_texts:
            # Tokenize with stride for long texts
            encodings = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_overflowing_tokens=True,
                stride=stride,
            )
            
            # Process each chunk
            for input_ids, attention_mask in zip(encodings["input_ids"], encodings["attention_mask"]):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                # Create labels (shift right)
                labels = input_ids.clone()
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        labels=labels.unsqueeze(0),
                    )
                
                # Get loss
                loss = outputs.loss.item()
                
                # Count non-padding tokens
                num_tokens = attention_mask.sum().item()
                
                batch_loss += loss * num_tokens
                batch_tokens += num_tokens
        
        total_loss += batch_loss
        total_tokens += batch_tokens
    
    # Calculate perplexity and other metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    metrics = {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
    }
    
    logger.info(f"Perplexity evaluation complete: {perplexity:.4f}")
    
    return metrics


def evaluate_with_lm_harness(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: List[str] = None,
    batch_size: int = 8,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate model using the LM Evaluation Harness.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        tasks: List of tasks to evaluate on (default: popular benchmarks)
        batch_size: Batch size for evaluation
        output_path: Path to save the results
        
    Returns:
        Dict[str, Any]: Evaluation results by task
    """
    try:
        from lm_eval import evaluator, tasks as lm_tasks
    except ImportError:
        logger.error(
            "lm-evaluation-harness not found. Install with: "
            "pip install lm-evaluation-harness"
        )
        return {"error": "lm-evaluation-harness not installed"}
    
    logger.info("Running LM Evaluation Harness...")
    
    # Default to a standard set of tasks if none specified
    if tasks is None:
        tasks = [
            "hellaswag",
            "mmlu",
            "truthfulqa_mc",
            "winogrande",
            "gsm8k",
            "arc_challenge",
        ]
    
    # Create model adapter for lm-eval
    def model_adapter(batch_inputs):
        """Adapter function for lm-eval-harness."""
        input_ids = [torch.tensor(inputs) for inputs in batch_inputs]
        # Pad to max length in batch
        max_length = max(len(ids) for ids in input_ids)
        padded_ids = [
            torch.cat([ids, torch.zeros(max_length - len(ids), dtype=torch.long)])
            for ids in input_ids
        ]
        padded_ids = torch.stack(padded_ids).to(model.device)
        
        # Create attention mask
        attention_mask = torch.zeros_like(padded_ids)
        for i, ids in enumerate(input_ids):
            attention_mask[i, :len(ids)] = 1
        
        # Run model
        with torch.no_grad():
            outputs = model(input_ids=padded_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Extract relevant logits (last token)
        results = []
        for i, ids in enumerate(input_ids):
            results.append(logits[i, :len(ids)])
        
        return results
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=model_adapter,
        tasks=tasks,
        batch_size=batch_size,
    )
    
    # Save results if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    return results


def generate_outputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    prompt_column: str,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    generation_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate outputs from the model for a dataset of prompts.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        dataset: Dataset containing prompts
        prompt_column: Column name for prompts
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for generation
        generation_config: Configuration for generation
        
    Returns:
        List[Dict[str, Any]]: Generated outputs
    """
    logger.info("Generating outputs for evaluation...")
    
    # Set up default generation config
    if generation_config is None:
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    
    # Create a generation pipeline
    generation_config = GenerationConfig(**generation_config)
    pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=model.device,
        generation_config=generation_config,
    )
    
    # Prepare prompts
    prompts = dataset[prompt_column]
    
    # Generate in batches
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        results = pipeline(batch_prompts)
        
        # Flatten results if needed
        if not isinstance(results[0], list):
            batch_results = [results]
        else:
            batch_results = results
        
        # Store outputs with prompt and metadata
        for j, result in enumerate(batch_results):
            prompt = batch_prompts[j if j < len(batch_prompts) else 0]
            
            # Store all generations for this prompt
            for gen in result:
                outputs.append({
                    "prompt": prompt,
                    "generated_text": gen["generated_text"],
                    "full_text": gen["generated_text"],
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "generated_tokens": len(tokenizer.encode(gen["generated_text"])) - len(tokenizer.encode(prompt)),
                })
    
    logger.info(f"Generated {len(outputs)} outputs for evaluation")
    return outputs


def calculate_text_metrics(
    references: List[str],
    predictions: List[str],
) -> Dict[str, float]:
    """
    Calculate various text similarity metrics between references and predictions.
    
    Args:
        references: List of reference texts
        predictions: List of generated texts
        
    Returns:
        Dict[str, float]: Metrics including BLEU, ROUGE, etc.
    """
    logger.info("Calculating text similarity metrics...")
    
    # Tokenize for BLEU calculation
    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_preds = [pred.split() for pred in predictions]
    
    # Set up ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate BLEU
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothing)
    
    # Calculate ROUGE for each pair and average
    rouge_scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
    }
    
    for ref, pred in zip(references, predictions):
        scores = rouge.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    
    # Average ROUGE scores
    total = len(references)
    for key in rouge_scores:
        rouge_scores[key] /= total if total > 0 else 1
    
    # Combine all metrics
    metrics = {
        'bleu': bleu_score,
        'rouge1_f': rouge_scores['rouge1'],
        'rouge2_f': rouge_scores['rouge2'],
        'rougeL_f': rouge_scores['rougeL'],
    }
    
    return metrics


def prepare_human_eval_interface(
    model_outputs: List[Dict[str, Any]],
    model_name: str,
    output_path: str,
) -> None:
    """
    Prepare a human evaluation interface by formatting model outputs.
    
    Args:
        model_outputs: List of model generation outputs
        model_name: Name of the model
        output_path: Path to save the prepared evaluation data
    """
    logger.info(f"Preparing human evaluation interface for {model_name}...")
    
    # Format data for human evaluation
    eval_data = []
    for i, output in enumerate(model_outputs):
        eval_data.append({
            "id": i,
            "model": model_name,
            "prompt": output["prompt"],
            "response": output["generated_text"],
            "promptID": f"prompt_{i}",
            "score_relevance": None,
            "score_accuracy": None,
            "score_coherence": None,
            "score_overall": None,
            "evaluator_comments": "",
        })
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON and CSV for flexibility
    with open(output_path + ".json", "w") as f:
        json.dump(eval_data, f, indent=2)
        
    # Create CSV for spreadsheet evaluations
    df = pd.DataFrame(eval_data)
    df.to_csv(output_path + ".csv", index=False)
    
    logger.info(f"Human evaluation data prepared and saved to {output_path}")


def load_human_eval_results(
    eval_path: str,
) -> Dict[str, Any]:
    """
    Load and analyze human evaluation results.
    
    Args:
        eval_path: Path to human evaluation results file
        
    Returns:
        Dict[str, Any]: Analysis of human evaluation scores
    """
    logger.info(f"Loading human evaluation results from {eval_path}...")
    
    # Check file extension and load accordingly
    if eval_path.endswith(".csv"):
        df = pd.read_csv(eval_path)
        eval_data = df.to_dict(orient="records")
    elif eval_path.endswith(".json"):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {eval_path}")
    
    # Extract scores
    scores = {
        "relevance": [],
        "accuracy": [],
        "coherence": [],
        "overall": [],
    }
    
    for item in eval_data:
        if item["score_relevance"] is not None:
            scores["relevance"].append(item["score_relevance"])
        if item["score_accuracy"] is not None:
            scores["accuracy"].append(item["score_accuracy"])
        if item["score_coherence"] is not None:
            scores["coherence"].append(item["score_coherence"])
        if item["score_overall"] is not None:
            scores["overall"].append(item["score_overall"])
    
    # Calculate statistics
    results = {}
    for category, values in scores.items():
        if values:
            results[f"{category}_mean"] = sum(values) / len(values)
            results[f"{category}_median"] = sorted(values)[len(values) // 2]
            results[f"{category}_min"] = min(values)
            results[f"{category}_max"] = max(values)
            results[f"{category}_std"] = np.std(values)
            results[f"{category}_count"] = len(values)
    
    return results


def run_domain_specific_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    domain: str,
    evaluation_data_path: str,
    batch_size: int = 4,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run domain-specific evaluation.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        domain: Domain for evaluation (e.g., "medical", "legal", "code")
        evaluation_data_path: Path to evaluation data
        batch_size: Batch size for evaluation
        output_path: Path to save the evaluation results
        
    Returns:
        Dict[str, Any]: Domain-specific evaluation metrics
    """
    logger.info(f"Running domain-specific evaluation for {domain}...")
    
    # Load evaluation data
    if evaluation_data_path.endswith(".json"):
        with open(evaluation_data_path, "r") as f:
            eval_data = json.load(f)
            
        if isinstance(eval_data, dict):
            # Handle different formats
            if "data" in eval_data:
                eval_data = eval_data["data"]
            elif "examples" in eval_data:
                eval_data = eval_data["examples"]
    
    elif evaluation_data_path.endswith(".csv"):
        df = pd.read_csv(evaluation_data_path)
        eval_data = df.to_dict(orient="records")
    
    else:
        raise ValueError(f"Unsupported file format: {evaluation_data_path}")
    
    # Handle domain-specific evaluation
    if domain == "medical":
        metrics = evaluate_medical_domain(model, tokenizer, eval_data, batch_size)
    
    elif domain == "legal":
        metrics = evaluate_legal_domain(model, tokenizer, eval_data, batch_size)
    
    elif domain == "code":
        metrics = evaluate_code_domain(model, tokenizer, eval_data, batch_size)
    
    elif domain == "reasoning":
        metrics = evaluate_reasoning_domain(model, tokenizer, eval_data, batch_size)
    
    else:
        # Generic domain evaluation
        metrics = evaluate_generic_domain(model, tokenizer, eval_data, batch_size)
    
    # Save results if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


def evaluate_generic_domain(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: List[Dict[str, Any]],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Generic domain evaluation.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        eval_data: Evaluation data
        batch_size: Batch size for evaluation
        
    Returns:
        Dict[str, Any]: Evaluation metrics
    """
    # Extract prompts and references
    prompts = []
    references = []
    
    for item in eval_data:
        # Handle different data formats
        if "prompt" in item and "reference" in item:
            prompts.append(item["prompt"])
            references.append(item["reference"])
        elif "input" in item and "output" in item:
            prompts.append(item["input"])
            references.append(item["output"])
        elif "question" in item and "answer" in item:
            prompts.append(item["question"])
            references.append(item["answer"])
    
    # Generate outputs
    outputs = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        dataset={"text": prompts},
        prompt_column="text",
        batch_size=batch_size,
    )
    
    # Extract generated texts
    predictions = [output["generated_text"] for output in outputs]
    
    # Calculate metrics
    metrics = calculate_text_metrics(references, predictions)
    
    return metrics


def evaluate_medical_domain(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: List[Dict[str, Any]],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Medical domain evaluation.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        eval_data: Evaluation data
        batch_size: Batch size for evaluation
        
    Returns:
        Dict[str, Any]: Medical domain evaluation metrics
    """
    # Implementation of medical-specific evaluation
    # TODO: Add specialized metrics for medical domain
    return evaluate_generic_domain(model, tokenizer, eval_data, batch_size)


def evaluate_legal_domain(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: List[Dict[str, Any]],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Legal domain evaluation.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        eval_data: Evaluation data
        batch_size: Batch size for evaluation
        
    Returns:
        Dict[str, Any]: Legal domain evaluation metrics
    """
    # Implementation of legal-specific evaluation
    # TODO: Add specialized metrics for legal domain
    return evaluate_generic_domain(model, tokenizer, eval_data, batch_size)


def evaluate_code_domain(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: List[Dict[str, Any]],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Code domain evaluation.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        eval_data: Evaluation data
        batch_size: Batch size for evaluation
        
    Returns:
        Dict[str, Any]: Code domain evaluation metrics
    """
    # Implementation of code-specific evaluation
    # TODO: Add specialized metrics for code evaluation
    return evaluate_generic_domain(model, tokenizer, eval_data, batch_size)


def evaluate_reasoning_domain(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: List[Dict[str, Any]],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Reasoning domain evaluation.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        eval_data: Evaluation data
        batch_size: Batch size for evaluation
        
    Returns:
        Dict[str, Any]: Reasoning domain evaluation metrics
    """
    # Implementation of reasoning-specific evaluation
    # TODO: Add specialized metrics for reasoning evaluation
    return evaluate_generic_domain(model, tokenizer, eval_data, batch_size)


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Optional[Dataset] = None,
    benchmarks: List[str] = None,
    batch_size: int = 8,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation using multiple methods.
    
    Args:
        model: Pretrained model
        tokenizer: Tokenizer for the model
        eval_dataset: Optional evaluation dataset
        benchmarks: List of benchmarks to run
        batch_size: Batch size for evaluation
        output_dir: Directory to save evaluation results
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    logger.info("Starting comprehensive model evaluation...")
    
    results = {}
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run perplexity evaluation if dataset provided
    if eval_dataset is not None:
        logger.info("Evaluating perplexity...")
        perplexity_results = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            batch_size=batch_size,
        )
        results["perplexity"] = perplexity_results
        
        # Save perplexity results
        if output_dir:
            with open(os.path.join(output_dir, "perplexity_results.json"), "w") as f:
                json.dump(perplexity_results, f, indent=2)
    
    # Run benchmark evaluations
    if benchmarks:
        logger.info(f"Running benchmark evaluations: {benchmarks}")
        
        if "lm-evaluation-harness" in benchmarks:
            lm_eval_results = evaluate_with_lm_harness(
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                output_path=os.path.join(output_dir, "lm_eval_results.json") if output_dir else None,
            )
            results["lm_eval_harness"] = lm_eval_results
        
        if "domain-specific-eval" in benchmarks:
            # Run domain-specific evaluations (customize as needed)
            domains = ["medical", "legal", "code", "reasoning"]
            domain_results = {}
            
            for domain in domains:
                # Check if domain-specific data exists
                data_path = f"evaluation/domain_{domain}.json"
                if not os.path.exists(data_path):
                    logger.warning(f"Domain evaluation data not found: {data_path}")
                    continue
                
                # Run domain evaluation
                domain_eval_results = run_domain_specific_evaluation(
                    model=model,
                    tokenizer=tokenizer,
                    domain=domain,
                    evaluation_data_path=data_path,
                    batch_size=batch_size,
                    output_path=os.path.join(output_dir, f"domain_{domain}_results.json") if output_dir else None,
                )
                domain_results[domain] = domain_eval_results
            
            results["domain_specific"] = domain_results
        
        if "human-eval" in benchmarks and eval_dataset is not None:
            # Prepare data for human evaluation
            logger.info("Preparing data for human evaluation...")
            
            # Generate some outputs for human evaluation
            num_samples = min(50, len(eval_dataset))
            sample_indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
            sample_dataset = eval_dataset.select(sample_indices)
            
            # Get prompts column
            prompt_column = None
            for col in ["instruction", "prompt", "input", "query"]:
                if col in sample_dataset.column_names:
                    prompt_column = col
                    break
            
            if prompt_column is None:
                logger.warning("Could not identify prompt column for human evaluation")
            else:
                # Generate outputs
                model_outputs = generate_outputs(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=sample_dataset,
                    prompt_column=prompt_column,
                    batch_size=batch_size,
                )
                
                # Prepare human evaluation interface
                model_name = model.config.name_or_path.split("/")[-1]
                human_eval_path = os.path.join(output_dir, "human_eval_data") if output_dir else "human_eval_data"
                prepare_human_eval_interface(
                    model_outputs=model_outputs,
                    model_name=model_name,
                    output_path=human_eval_path,
                )
                
                results["human_eval_prepared"] = {
                    "samples": num_samples,
                    "output_path": human_eval_path,
                }
    
    logger.info("Model evaluation complete")
    return results
