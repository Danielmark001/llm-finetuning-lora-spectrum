#!/usr/bin/env python
"""
Comprehensive benchmarking script for Artemis models.

This script provides utilities to benchmark Artemis models across various
dimensions including:
- Performance (latency, throughput)
- Memory usage
- Accuracy on custom benchmarks
- Domain-specific metrics
- Efficiency across different hardware
"""

import argparse
import json
import logging
import os
import sys
import time
import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from tqdm import tqdm

# Add parent directory to path to allow importing from artemis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define benchmark types
BENCHMARK_TYPES = [
    "performance",
    "memory",
    "accuracy",
    "domain",
    "hardware",
    "all"
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Artemis Model Benchmarking Tool")
    
    # Model specification
    parser.add_argument(
        "--model_path", 
        type=str,
        required=True,
        help="Path to model directory or checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["medical", "legal", "multilingual", "custom"],
        default="custom",
        help="Type of Artemis model"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model configuration YAML file (optional)"
    )
    
    # Benchmark specification
    parser.add_argument(
        "--benchmark_type",
        type=str,
        choices=BENCHMARK_TYPES,
        default="all",
        help="Type of benchmark to run"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to benchmark dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "csv", "md", "all"],
        default="all",
        help="Format for results output"
    )
    
    # Performance benchmark options
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated list of batch sizes for performance benchmarks"
    )
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="128,256,512,1024",
        help="Comma-separated list of sequence lengths for performance benchmarks"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations for performance benchmarks"
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations before measurement"
    )
    
    # Hardware benchmark options
    parser.add_argument(
        "--devices",
        type=str,
        default="cpu",
        help="Comma-separated list of devices to benchmark on (cpu, cuda:0, etc.)"
    )
    
    # Comparison options
    parser.add_argument(
        "--compare_with",
        type=str,
        help="Path to another model to compare with"
    )
    parser.add_argument(
        "--baseline_results",
        type=str,
        help="Path to baseline results JSON for comparison"
    )
    
    # Additional options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed profiling (PyTorch profiler)"
    )
    parser.add_argument(
        "--save_traces",
        action="store_true",
        help="Save profiler traces for visualization"
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(model_path, device="cuda"):
    """
    Load model and tokenizer.
    
    Args:
        model_path: Path to model
        device: Device to load model on
        
    Returns:
        tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading model from {model_path}")
    
    # Check for CUDA availability if requested
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        )
    
    logger.info(f"Model loaded on {device}")
    
    return model, tokenizer

def load_benchmark_dataset(args):
    """
    Load benchmark dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        Benchmark dataset
    """
    if not args.dataset_path:
        raise ValueError("Dataset path must be provided for accuracy benchmarks")
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load dataset based on file extension
    if dataset_path.suffix == ".json":
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    
    elif dataset_path.suffix == ".jsonl":
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    
    elif dataset_path.suffix in [".csv", ".tsv"]:
        sep = "," if dataset_path.suffix == ".csv" else "\t"
        dataset = pd.read_csv(dataset_path, sep=sep)
        dataset = dataset.to_dict('records')
    
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    
    logger.info(f"Loaded benchmark dataset with {len(dataset)} samples")
    
    return dataset

def run_performance_benchmark(model, tokenizer, args):
    """
    Run performance benchmark measuring latency and throughput.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Running performance benchmark")
    
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    sequence_lengths = [int(sl) for sl in args.sequence_lengths.split(',')]
    
    results = {
        "latency_ms": {},
        "throughput_samples_per_second": {},
        "throughput_tokens_per_second": {},
        "batch_size_scaling_efficiency": {},
        "config": {
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "batch_sizes": batch_sizes,
            "sequence_lengths": sequence_lengths,
            "device": model.device
        }
    }
    
    # Sample input text for benchmarking
    sample_text = "This is a sample text for benchmarking. " * 10
    
    # Run benchmarks for each combination of batch size and sequence length
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            key = f"batch_{batch_size}_seq_{seq_length}"
            logger.info(f"Benchmarking with batch size {batch_size}, sequence length {seq_length}")
            
            # Create input batch
            input_ids_list = []
            attention_mask_list = []
            
            for _ in range(batch_size):
                # Tokenize and truncate/pad to the desired sequence length
                encoded = tokenizer.encode_plus(
                    sample_text,
                    max_length=seq_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids_list.append(encoded["input_ids"])
                attention_mask_list.append(encoded["attention_mask"])
            
            input_ids = torch.cat(input_ids_list, dim=0).to(model.device)
            attention_mask = torch.cat(attention_mask_list, dim=0).to(model.device)
            
            # Warmup runs
            logger.info(f"Running {args.warmup_iterations} warmup iterations...")
            for _ in range(args.warmup_iterations):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Benchmark runs
            logger.info(f"Running {args.iterations} benchmark iterations...")
            latencies = []
            
            # Use PyTorch profiler if requested
            if args.profile:
                from torch.profiler import profile, record_function, ProfilerActivity
                
                activities = [ProfilerActivity.CPU]
                if model.device.type == "cuda":
                    activities.append(ProfilerActivity.CUDA)
                
                with profile(
                    activities=activities,
                    record_shapes=True,
                ) as prof:
                    for i in range(args.iterations):
                        with record_function(f"iteration_{i}"):
                            start_time = time.time()
                            with torch.no_grad():
                                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                            end_time = time.time()
                            latencies.append((end_time - start_time) * 1000)  # ms
                
                # Save profile results if requested
                if args.save_traces:
                    os.makedirs(args.output_dir, exist_ok=True)
                    trace_path = Path(args.output_dir) / f"profile_{key}.json"
                    prof.export_chrome_trace(str(trace_path))
                    logger.info(f"Saved profiler trace to {trace_path}")
                
                # Add profiler summary to results
                results[f"profile_summary_{key}"] = str(prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=10
                ))
            
            else:
                # Regular benchmarking without profiler
                for _ in range(args.iterations):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    end_time = time.time()
                    latencies.append((end_time - start_time) * 1000)  # ms
            
            # Calculate statistics
            mean_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            throughput_samples = batch_size * (1000 / mean_latency)
            throughput_tokens = batch_size * seq_length * (1000 / mean_latency)
            
            # Store results
            results["latency_ms"][key] = {
                "mean": mean_latency,
                "p50": p50_latency,
                "p95": p95_latency,
                "p99": p99_latency,
                "min": min(latencies),
                "max": max(latencies),
                "std": np.std(latencies)
            }
            
            results["throughput_samples_per_second"][key] = throughput_samples
            results["throughput_tokens_per_second"][key] = throughput_tokens
    
    # Calculate batch size scaling efficiency
    for seq_length in sequence_lengths:
        scaling_efficiency = {}
        base_bs = batch_sizes[0]
        base_key = f"batch_{base_bs}_seq_{seq_length}"
        base_throughput = results["throughput_samples_per_second"][base_key]
        
        for bs in batch_sizes[1:]:
            key = f"batch_{bs}_seq_{seq_length}"
            current_throughput = results["throughput_samples_per_second"][key]
            scaling_factor = bs / base_bs
            ideal_throughput = base_throughput * scaling_factor
            efficiency = (current_throughput / ideal_throughput) * 100
            
            scaling_efficiency[key] = {
                "efficiency_percent": efficiency,
                "actual_throughput": current_throughput,
                "ideal_throughput": ideal_throughput
            }
        
        results["batch_size_scaling_efficiency"][f"seq_{seq_length}"] = scaling_efficiency
    
    logger.info("Performance benchmark completed")
    
    return results

def run_memory_benchmark(model, tokenizer, args):
    """
    Run memory usage benchmark.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Running memory benchmark")
    
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    sequence_lengths = [int(sl) for sl in args.sequence_lengths.split(',')]
    
    results = {
        "memory_usage_mb": {},
        "config": {
            "batch_sizes": batch_sizes,
            "sequence_lengths": sequence_lengths,
            "device": model.device
        }
    }
    
    # Sample input text for benchmarking
    sample_text = "This is a sample text for benchmarking. " * 10
    
    # Run benchmarks for each combination of batch size and sequence length
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            key = f"batch_{batch_size}_seq_{seq_length}"
            logger.info(f"Measuring memory with batch size {batch_size}, sequence length {seq_length}")
            
            # Clear CUDA cache if using GPU
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
                # Get initial memory usage
                memory_allocated_start = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                memory_reserved_start = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            else:
                # On CPU, we can't easily measure memory usage
                memory_allocated_start = 0
                memory_reserved_start = 0
            
            # Create input batch
            input_ids_list = []
            attention_mask_list = []
            
            for _ in range(batch_size):
                encoded = tokenizer.encode_plus(
                    sample_text,
                    max_length=seq_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids_list.append(encoded["input_ids"])
                attention_mask_list.append(encoded["attention_mask"])
            
            input_ids = torch.cat(input_ids_list, dim=0).to(model.device)
            attention_mask = torch.cat(attention_mask_list, dim=0).to(model.device)
            
            # Run the model to measure memory usage during inference
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get peak memory usage
            if model.device.type == "cuda":
                memory_allocated_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                memory_reserved_peak = torch.cuda.max_memory_reserved() / (1024 * 1024)  # MB
                
                # Calculate the difference from the start
                memory_allocated_diff = memory_allocated_peak - memory_allocated_start
                memory_reserved_diff = memory_reserved_peak - memory_reserved_start
            else:
                # On CPU, we can't easily measure memory usage
                memory_allocated_peak = "N/A"
                memory_reserved_peak = "N/A"
                memory_allocated_diff = "N/A"
                memory_reserved_diff = "N/A"
            
            # Store results
            results["memory_usage_mb"][key] = {
                "memory_allocated_start": memory_allocated_start,
                "memory_allocated_peak": memory_allocated_peak,
                "memory_allocated_diff": memory_allocated_diff,
                "memory_reserved_start": memory_reserved_start,
                "memory_reserved_peak": memory_reserved_peak,
                "memory_reserved_diff": memory_reserved_diff
            }
    
    logger.info("Memory benchmark completed")
    
    return results

def evaluate_accuracy(model, tokenizer, dataset, model_type, device="cuda"):
    """
    Evaluate model accuracy on the provided dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Benchmark dataset
        model_type: Type of the model (medical, legal, multilingual)
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Evaluating {model_type} model accuracy on dataset with {len(dataset)} samples")
    
    results = {
        "accuracy": 0.0,
        "per_domain_accuracy": {},
        "detailed_results": []
    }
    
    total_correct = 0
    domain_correct = {}
    domain_total = {}
    
    # Process each sample in the dataset
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        # Extract input and expected output based on model type
        if model_type == "medical":
            if "question" in sample:
                input_text = sample["question"]
            elif "query" in sample:
                input_text = sample["query"]
            else:
                input_text = sample["input"]
                
            if "reference_answer" in sample:
                expected_output = sample["reference_answer"]
            elif "answer" in sample:
                expected_output = sample["answer"]
            else:
                expected_output = sample["output"]
            
            # Get domain if available
            domain = sample.get("domain", "general")
            
        elif model_type == "legal":
            if "document" in sample and "query" in sample:
                input_text = f"Document: {sample['document']}\n\nQuery: {sample['query']}"
            else:
                input_text = sample["input"]
                
            expected_output = sample.get("answer", sample.get("output", ""))
            domain = sample.get("domain", sample.get("category", "general"))
            
        elif model_type == "multilingual":
            input_text = sample.get("query", sample.get("input", ""))
            expected_output = sample.get("answer", sample.get("output", ""))
            domain = sample.get("language", "en")
            
        else:  # custom
            input_text = sample.get("input", "")
            expected_output = sample.get("output", "")
            domain = sample.get("domain", "general")
        
        # Initialize domain counters if needed
        if domain not in domain_correct:
            domain_correct[domain] = 0
            domain_total[domain] = 0
        
        domain_total[domain] += 1
        
        # Generate model output
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_length=512,
                num_return_sequences=1,
                do_sample=False
            )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Evaluate correctness (simplified)
        # In a real implementation, this would use more sophisticated metrics
        # like F1 score, BLEU, ROUGE, exact match, etc.
        correct = evaluate_correctness(output_text, expected_output, model_type)
        
        if correct:
            total_correct += 1
            domain_correct[domain] += 1
        
        # Store detailed result
        results["detailed_results"].append({
            "input": input_text,
            "expected_output": expected_output,
            "model_output": output_text,
            "correct": correct,
            "domain": domain
        })
    
    # Calculate overall accuracy
    results["accuracy"] = total_correct / len(dataset) if len(dataset) > 0 else 0.0
    
    # Calculate per-domain accuracy
    for domain in domain_total.keys():
        domain_accuracy = domain_correct[domain] / domain_total[domain] if domain_total[domain] > 0 else 0.0
        results["per_domain_accuracy"][domain] = {
            "accuracy": domain_accuracy,
            "correct": domain_correct[domain],
            "total": domain_total[domain]
        }
    
    logger.info(f"Overall accuracy: {results['accuracy']:.4f}")
    
    return results

def evaluate_correctness(output_text, expected_output, model_type):
    """
    Evaluate if the model output is correct.
    
    Args:
        output_text: Model generated output
        expected_output: Expected output
        model_type: Type of the model
        
    Returns:
        Boolean indicating if the output is correct
    """
    from difflib import SequenceMatcher
    
    # This is a simplified evaluation function
    # In a real implementation, this would use more sophisticated metrics
    
    # Clean and normalize texts
    output_text = output_text.strip().lower()
    expected_output = expected_output.strip().lower()
    
    # Different evaluation methods based on model type
    if model_type == "medical":
        # For medical QA, check for overlap of key terms
        similarity = SequenceMatcher(None, output_text, expected_output).ratio()
        return similarity > 0.7
        
    elif model_type == "legal":
        # For legal analysis, check for exact matches of key points
        key_points_correct = all(point.lower() in output_text for point in expected_output.split('\n') if point.strip())
        return key_points_correct
        
    elif model_type == "multilingual":
        # For multilingual, check string similarity
        similarity = SequenceMatcher(None, output_text, expected_output).ratio()
        return similarity > 0.7
        
    else:  # custom
        # Default to string similarity
        similarity = SequenceMatcher(None, output_text, expected_output).ratio()
        return similarity > 0.8

def run_domain_benchmark(model, tokenizer, dataset, model_type, args):
    """
    Run domain-specific benchmarks.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        dataset: Benchmark dataset
        model_type: Type of the model
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Running domain-specific benchmark for {model_type} model")
    
    results = {
        "domain_metrics": {},
        "samples_evaluated": len(dataset)
    }
    
    # Different domain-specific metrics based on model type
    if model_type == "medical":
        # Medical-specific metrics
        results["domain_metrics"] = evaluate_medical_metrics(model, tokenizer, dataset)
        
    elif model_type == "legal":
        # Legal-specific metrics
        results["domain_metrics"] = evaluate_legal_metrics(model, tokenizer, dataset)
        
    elif model_type == "multilingual":
        # Multilingual-specific metrics
        results["domain_metrics"] = evaluate_multilingual_metrics(model, tokenizer, dataset)
        
    else:  # custom
        # General metrics
        results["domain_metrics"] = evaluate_general_metrics(model, tokenizer, dataset)
    
    logger.info("Domain benchmark completed")
    
    return results

def evaluate_medical_metrics(model, tokenizer, dataset):
    """
    Evaluate medical-specific metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Medical benchmark dataset
        
    Returns:
        Dictionary of medical-specific metrics
    """
    # Placeholder for actual implementation
    metrics = {
        "medical_relevance": 0.0,
        "evidence_support": 0.0,
        "diagnostic_accuracy": 0.0,
        "clinical_guidance_quality": 0.0,
        "bio_entity_recognition": 0.0
    }
    
    # In a real implementation, these would be calculated based on
    # domain-specific evaluation methods and external resources
    
    return metrics

def evaluate_legal_metrics(model, tokenizer, dataset):
    """
    Evaluate legal-specific metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Legal benchmark dataset
        
    Returns:
        Dictionary of legal-specific metrics
    """
    # Placeholder for actual implementation
    metrics = {
        "legal_reasoning": 0.0,
        "precedent_alignment": 0.0,
        "statutory_interpretation": 0.0,
        "case_law_citation": 0.0,
        "legal_entity_recognition": 0.0
    }
    
    # In a real implementation, these would be calculated based on
    # domain-specific evaluation methods and external resources
    
    return metrics

def evaluate_multilingual_metrics(model, tokenizer, dataset):
    """
    Evaluate multilingual-specific metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Multilingual benchmark dataset
        
    Returns:
        Dictionary of multilingual-specific metrics
    """
    # Group samples by language
    samples_by_language = {}
    
    for sample in dataset:
        language = sample.get("language", "en")
        if language not in samples_by_language:
            samples_by_language[language] = []
        samples_by_language[language].append(sample)
    
    # Calculate metrics per language
    metrics = {
        "cross_lingual_consistency": 0.0,
        "language_specific_performance": {},
        "cultural_sensitivity": 0.0,
        "terminology_accuracy": 0.0
    }
    
    # Calculate language-specific performance
    for language, samples in samples_by_language.items():
        # In a real implementation, these would be calculated based on
        # sophisticated evaluation methods for each language
        metrics["language_specific_performance"][language] = {
            "fluency": 0.0,
            "accuracy": 0.0,
            "idiomatic_correctness": 0.0,
            "sample_count": len(samples)
        }
    
    return metrics

def evaluate_general_metrics(model, tokenizer, dataset):
    """
    Evaluate general metrics for custom models.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Benchmark dataset
        
    Returns:
        Dictionary of general metrics
    """
    # Placeholder for actual implementation
    metrics = {
        "overall_quality": 0.0,
        "response_coherence": 0.0,
        "factual_accuracy": 0.0,
        "instruction_following": 0.0
    }
    
    # In a real implementation, these would be calculated based on
    # sophisticated evaluation methods
    
    return metrics

def run_hardware_benchmark(args):
    """
    Run benchmark across different hardware devices.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results across devices
    """
    logger.info("Running hardware benchmark")
    
    devices = args.devices.split(',')
    results = {
        "hardware_comparison": {},
        "devices_tested": devices
    }
    
    for device in devices:
        logger.info(f"Benchmarking on device: {device}")
        
        try:
            # Load model on the specific device
            model, tokenizer = load_model_and_tokenizer(args.model_path, device=device)
            
            # Run performance benchmark
            device_results = run_performance_benchmark(model, tokenizer, args)
            
            # Add to results
            results["hardware_comparison"][device] = device_results
            
            # Clean up to free memory
            del model
            torch.cuda.empty_cache() if 'cuda' in device else None
            
        except Exception as e:
            logger.error(f"Error benchmarking on device {device}: {e}")
            results["hardware_comparison"][device] = {"error": str(e)}
    
    logger.info("Hardware benchmark completed")
    
    return results

def save_benchmark_results(benchmark_results, args):
    """
    Save benchmark results in the specified format(s).
    
    Args:
        benchmark_results: Dictionary of benchmark results
        args: Command line arguments
    """
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"benchmark_{args.model_type}_{timestamp}"
    
    # Add metadata
    benchmark_results["metadata"] = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "benchmark_type": args.benchmark_type,
        "timestamp": timestamp,
        "args": vars(args)
    }
    
    # Save in JSON format
    if args.output_format in ["json", "all"]:
        json_path = Path(args.output_dir) / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Saved JSON results to {json_path}")
    
    # Save in CSV format
    if args.output_format in ["csv", "all"]:
        # Convert nested dictionaries to flat CSV files
        for key, value in benchmark_results.items():
            if isinstance(value, dict) and key != "metadata":
                csv_path = Path(args.output_dir) / f"{base_filename}_{key}.csv"
                
                try:
                    if key in ["latency_ms", "memory_usage_mb"]:
                        # These are particularly suited for CSV format
                        df = pd.json_normalize(value)
                        df.to_csv(csv_path, index=False)
                        logger.info(f"Saved CSV results to {csv_path}")
                except Exception as e:
                    logger.warning(f"Could not convert {key} to CSV: {e}")
    
    # Save in Markdown format
    if args.output_format in ["md", "all"]:
        md_path = Path(args.output_dir) / f"{base_filename}.md"
        
        with open(md_path, 'w') as f:
            f.write(f"# Artemis Benchmark Results\n\n")
            f.write(f"- **Model**: {args.model_path}\n")
            f.write(f"- **Model Type**: {args.model_type}\n")
            f.write(f"- **Benchmark Type**: {args.benchmark_type}\n")
            f.write(f"- **Date**: {timestamp}\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            if "performance" in benchmark_results:
                f.write("### Performance\n\n")
                f.write("| Batch Size | Sequence Length | Mean Latency (ms) | Throughput (samples/s) |\n")
                f.write("|------------|----------------|-------------------|------------------------|\n")
                
                for key in benchmark_results["performance"]["latency_ms"]:
                    parts = key.split("_")
                    batch_size = parts[1]
                    seq_length = parts[3]
                    latency = benchmark_results["performance"]["latency_ms"][key]["mean"]
                    throughput = benchmark_results["performance"]["throughput_samples_per_second"][key]
                    
                    f.write(f"| {batch_size} | {seq_length} | {latency:.2f} | {throughput:.2f} |\n")
            
            if "accuracy" in benchmark_results:
                f.write("\n### Accuracy\n\n")
                f.write(f"Overall Accuracy: {benchmark_results['accuracy']['accuracy']:.4f}\n\n")
                
                f.write("| Domain | Accuracy | Samples |\n")
                f.write("|--------|----------|----------|\n")
                
                for domain, metrics in benchmark_results["accuracy"]["per_domain_accuracy"].items():
                    f.write(f"| {domain} | {metrics['accuracy']:.4f} | {metrics['total']} |\n")
            
        logger.info(f"Saved Markdown results to {md_path}")
    
    logger.info(f"All benchmark results saved to {args.output_dir}")

def compare_with_baseline(benchmark_results, baseline_path):
    """
    Compare benchmark results with a baseline.
    
    Args:
        benchmark_results: Dictionary of current benchmark results
        baseline_path: Path to baseline results JSON
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing with baseline results from {baseline_path}")
    
    comparison = {
        "performance_diff": {},
        "accuracy_diff": {},
        "memory_diff": {},
        "summary": {}
    }
    
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        # Compare performance if available
        if "performance" in benchmark_results and "performance" in baseline:
            for key in benchmark_results["performance"]["latency_ms"]:
                if key in baseline["performance"]["latency_ms"]:
                    current_latency = benchmark_results["performance"]["latency_ms"][key]["mean"]
                    baseline_latency = baseline["performance"]["latency_ms"][key]["mean"]
                    
                    latency_diff = current_latency - baseline_latency
                    latency_diff_percent = (latency_diff / baseline_latency) * 100
                    
                    comparison["performance_diff"][key] = {
                        "latency_diff_ms": latency_diff,
                        "latency_diff_percent": latency_diff_percent,
                        "current_latency_ms": current_latency,
                        "baseline_latency_ms": baseline_latency,
                        "improved": latency_diff < 0
                    }
        
        # Compare accuracy if available
        if "accuracy" in benchmark_results and "accuracy" in baseline:
            current_acc = benchmark_results["accuracy"]["accuracy"]
            baseline_acc = baseline["accuracy"]["accuracy"]
            
            acc_diff = current_acc - baseline_acc
            acc_diff_percent = (acc_diff / baseline_acc) * 100 if baseline_acc > 0 else 0
            
            comparison["accuracy_diff"]["overall"] = {
                "accuracy_diff": acc_diff,
                "accuracy_diff_percent": acc_diff_percent,
                "current_accuracy": current_acc,
                "baseline_accuracy": baseline_acc,
                "improved": acc_diff > 0
            }
        
        # Create summary
        summary = {
            "overall_performance_change": "N/A",
            "overall_accuracy_change": "N/A",
            "overall_memory_change": "N/A"
        }
        
        if comparison["performance_diff"]:
            avg_latency_diff_percent = np.mean([v["latency_diff_percent"] for v in comparison["performance_diff"].values()])
            summary["overall_performance_change"] = f"{-avg_latency_diff_percent:.2f}%" if avg_latency_diff_percent < 0 else f"+{avg_latency_diff_percent:.2f}%"
        
        if "overall" in comparison["accuracy_diff"]:
            acc_diff_percent = comparison["accuracy_diff"]["overall"]["accuracy_diff_percent"]
            summary["overall_accuracy_change"] = f"+{acc_diff_percent:.2f}%" if acc_diff_percent > 0 else f"{acc_diff_percent:.2f}%"
        
        comparison["summary"] = summary
        
        logger.info(f"Comparison summary: {summary}")
        
    except Exception as e:
        logger.error(f"Error comparing with baseline: {e}")
        comparison["error"] = str(e)
    
    return comparison

def main():
    """Main function to run benchmarks."""
    args = parse_arguments()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        benchmark_results = {}
        
        # Run hardware benchmark separately as it loads the model multiple times
        if args.benchmark_type in ["hardware", "all"]:
            benchmark_results["hardware"] = run_hardware_benchmark(args)
            
            # If only hardware benchmark was requested, save and exit
            if args.benchmark_type == "hardware":
                save_benchmark_results(benchmark_results, args)
                return
        
        # For other benchmark types, load the model once
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # Run performance benchmark
        if args.benchmark_type in ["performance", "all"]:
            benchmark_results["performance"] = run_performance_benchmark(model, tokenizer, args)
        
        # Run memory benchmark
        if args.benchmark_type in ["memory", "all"]:
            benchmark_results["memory"] = run_memory_benchmark(model, tokenizer, args)
        
        # Run accuracy benchmark
        if args.benchmark_type in ["accuracy", "all", "domain"]:
            # Load dataset
            dataset = load_benchmark_dataset(args)
            
            if args.benchmark_type in ["accuracy", "all"]:
                benchmark_results["accuracy"] = evaluate_accuracy(model, tokenizer, dataset, args.model_type)
            
            # Run domain-specific benchmark
            if args.benchmark_type in ["domain", "all"]:
                benchmark_results["domain"] = run_domain_benchmark(model, tokenizer, dataset, args.model_type, args)
        
        # Save results
        save_benchmark_results(benchmark_results, args)
        
        # Compare with baseline if provided
        if args.baseline_results:
            comparison = compare_with_baseline(benchmark_results, args.baseline_results)
            
            # Save comparison results
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            comparison_path = Path(args.output_dir) / f"comparison_{timestamp}.json"
            
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logger.info(f"Saved comparison results to {comparison_path}")
        
        logger.info("Benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
