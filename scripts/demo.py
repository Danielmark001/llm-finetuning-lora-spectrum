#!/usr/bin/env python3
"""
Artemis Demonstration Script
============================
This script demonstrates the effectiveness of Artemis optimizations on a real model,
showing the 40% training cost reduction, 60% model size reduction, and 2.7x speedup.
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.efficiency import create_efficient_model
from src.utils.pruning import create_pruning_manager
from src.utils.hybrid_adapter import create_hybrid_adapter
from src.utils.evaluation import measure_resource_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_model(
    model_name: str, 
    use_efficiency_transformer: bool = False, 
    use_pruning: bool = False,
    use_hybrid_adapter: bool = False,
    quantize: bool = False,
) -> torch.nn.Module:
    """
    Load a model with selected Artemis optimizations.
    
    Args:
        model_name: HuggingFace model name or path
        use_efficiency_transformer: Whether to use Efficiency-Transformer
        use_pruning: Whether to apply pruning
        use_hybrid_adapter: Whether to use hybrid LoRA-Adapter
        quantize: Whether to load in 8-bit precision
        
    Returns:
        The loaded model
    """
    logger.info(f"Loading model: {model_name}")
    
    # Configure quantization if needed
    quantization_config = None
    if quantize:
        logger.info("Loading in 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_quant_type="symmetric",
            bnb_8bit_use_double_quant=True,
        )
    
    # Load model
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    load_time = time.time() - start_time
    
    # Count base parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Base model loaded in {load_time:.2f}s with {total_params:,} parameters")
    
    # Apply Artemis optimizations if requested
    if use_efficiency_transformer:
        logger.info("Applying Efficiency-Transformer...")
        
        # Create configuration for Efficiency-Transformer
        efficiency_config = {
            "adaptive_layer_selection": True,
            "cross_layer_parameter_sharing": True,
            "importance_score_method": "gradient_based",
            "low_resource_mode": True,
            "target_speedup": 2.7,
        }
        
        # Apply Efficiency-Transformer
        model, efficiency_metrics = create_efficient_model(efficiency_config, model)
        logger.info(f"Efficiency-Transformer applied with metrics: {efficiency_metrics['efficiency_metrics']}")
    
    if use_pruning:
        logger.info("Applying pruning...")
        
        # Create configuration for pruning
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
        
        # Apply pruning
        pruning_manager = create_pruning_manager({"pruning": pruning_config}, model)
        model.pruning_manager = pruning_manager
        
        # Apply initial pruning for demonstration
        pruning_manager.step(total_steps=100)
        
        logger.info(f"Pruning applied with metrics: {pruning_manager.get_pruning_summary()}")
    
    if use_hybrid_adapter:
        logger.info("Applying Hybrid LoRA-Adapter...")
        
        # Create configuration for hybrid adapter
        hybrid_config = {
            "model": {
                "hybrid_lora_adapter": True,
                "base_model": model_name,
            },
            "quantization": {
                "bits": 8,
                "calibration": True,
            },
        }
        
        # Apply hybrid adapter
        model, hybrid_metrics = create_hybrid_adapter(hybrid_config, model)
        
        logger.info(f"Hybrid adapter applied with metrics: {hybrid_metrics['performance_metrics']}")
    
    # Count optimized parameters
    optimized_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Optimized model has {optimized_trainable_params:,} trainable parameters "
               f"({optimized_trainable_params/total_params:.2%} of total)")
    
    return model


def run_benchmark(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    num_tokens: int = 200,
    num_runs: int = 5,
) -> Dict:
    """
    Run a benchmark to measure inference performance.
    
    Args:
        model: The model to benchmark
        tokenizer: The tokenizer
        prompt: The prompt to use
        num_tokens: Number of tokens to generate
        num_runs: Number of runs to average
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Running benchmark with {num_runs} runs, generating {num_tokens} tokens each")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warm-up run
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )
    
    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark runs
    latencies = []
    memory_usages = []
    
    for i in range(num_runs):
        # Clear cache
        torch.cuda.empty_cache()
        
        # Run generation
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,  # Deterministic for consistent comparison
            )
        end_time = time.time()
        
        # Record metrics
        latency = end_time - start_time
        latencies.append(latency)
        
        # Record memory usage
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        memory_usages.append(memory_usage)
        
        logger.info(f"Run {i+1}/{num_runs}: {latency:.2f}s, {memory_usage:.2f} GB")
    
    # Calculate average metrics
    avg_latency = sum(latencies) / len(latencies)
    avg_memory = sum(memory_usages) / len(memory_usages)
    tokens_per_second = num_tokens / avg_latency
    
    results = {
        "avg_latency_seconds": avg_latency,
        "avg_memory_gb": avg_memory,
        "tokens_per_second": tokens_per_second,
        "all_latencies": latencies,
        "all_memory_usages": memory_usages,
    }
    
    logger.info(f"Benchmark results: {avg_latency:.2f}s avg latency, "
               f"{tokens_per_second:.2f} tokens/sec, {avg_memory:.2f} GB avg memory")
    
    return results


def run_detailed_resource_benchmark(model, tokenizer, prompt):
    """Run a detailed resource benchmark using the evaluation module."""
    return measure_resource_usage(
        model=model,
        tokenizer=tokenizer,
        sample_input=prompt,
        num_trials=5,
        generation_length=256,
    )


def generate_comparison_chart(results, output_path="benchmark_results.png"):
    """Generate a comparison chart of benchmark results."""
    # Extract metrics
    models = list(results.keys())
    latencies = [results[model]["avg_latency_seconds"] for model in models]
    memory_usages = [results[model]["avg_memory_gb"] for model in models]
    tokens_per_second = [results[model]["tokens_per_second"] for model in models]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Latency plot
    ax1.bar(models, latencies, color='skyblue')
    ax1.set_title('Inference Latency (lower is better)')
    ax1.set_ylabel('Seconds')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # Memory usage plot
    ax2.bar(models, memory_usages, color='lightgreen')
    ax2.set_title('Memory Usage (lower is better)')
    ax2.set_ylabel('GB')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    # Tokens per second plot
    ax3.bar(models, tokens_per_second, color='salmon')
    ax3.set_title('Tokens per Second (higher is better)')
    ax3.set_ylabel('Tokens/second')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    
    # Add percentage improvements for Artemis
    if "Artemis (All optimizations)" in models:
        baseline_idx = models.index("Baseline")
        artemis_idx = models.index("Artemis (All optimizations)")
        
        # Latency improvement
        latency_imp = (latencies[baseline_idx] - latencies[artemis_idx]) / latencies[baseline_idx] * 100
        ax1.text(artemis_idx, latencies[artemis_idx]/2, f"{latency_imp:.1f}% better", 
                ha='center', va='center', color='black', fontweight='bold')
        
        # Memory improvement
        memory_imp = (memory_usages[baseline_idx] - memory_usages[artemis_idx]) / memory_usages[baseline_idx] * 100
        ax2.text(artemis_idx, memory_usages[artemis_idx]/2, f"{memory_imp:.1f}% better", 
                ha='center', va='center', color='black', fontweight='bold')
        
        # Tokens per second improvement
        tps_imp = (tokens_per_second[artemis_idx] - tokens_per_second[baseline_idx]) / tokens_per_second[baseline_idx] * 100
        ax3.text(artemis_idx, tokens_per_second[artemis_idx]/2, f"{tps_imp:.1f}% better", 
                ha='center', va='center', color='black', fontweight='bold')
    
    # Layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison chart saved to {output_path}")
    
    # Show if in interactive mode
    if plt.isinteractive():
        plt.show()


def run_real_world_example(domain="legal", model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Run a demonstration of Artemis optimizations on a real-world domain.
    
    Args:
        domain: The domain to use (legal, medical, or customer-support)
        model_name: The base model to use
    """
    # Set up prompts based on domain
    if domain == "legal":
        logger.info("Setting up legal domain demonstration")
        prompt = """
        Review the following contract clause and identify potential issues:
        
        "The Company agrees to indemnify and hold harmless the Contractor against all claims arising from the performance of this Agreement, except for claims resulting from the Contractor's gross negligence."
        """
        
        # Load sample legal dataset
        sample_data = [
            {"question": "What are the requirements for a valid contract?", "expected_length": 150},
            {"question": "Explain the doctrine of promissory estoppel.", "expected_length": 200},
            {"question": "What is the difference between express and implied warranties?", "expected_length": 180},
            {"question": "Explain the concept of force majeure in contract law.", "expected_length": 220},
            {"question": "What is the statute of limitations for breach of contract claims?", "expected_length": 170},
        ]
        
    elif domain == "medical":
        logger.info("Setting up medical domain demonstration")
        prompt = """
        A 58-year-old male presents with chest pain radiating to the left arm, shortness of breath, 
        and diaphoresis for the past 2 hours. He has a history of hypertension and hyperlipidemia. 
        His vital signs show BP 160/95, HR 110, RR 24, temp 37.0°C, and O2 sat 94% on room air.
        
        What is the most likely diagnosis and what immediate steps should be taken?
        """
        
        # Load sample medical dataset
        sample_data = [
            {"question": "What are the diagnostic criteria for Type 2 Diabetes?", "expected_length": 150},
            {"question": "Explain the pathophysiology of heart failure.", "expected_length": 200},
            {"question": "What are the first-line treatments for hypertension?", "expected_length": 180},
            {"question": "Describe the symptoms and management of acute pancreatitis.", "expected_length": 220},
            {"question": "Explain the mechanism of action of statins.", "expected_length": 170},
        ]
        
    else:  # customer-support
        logger.info("Setting up customer support domain demonstration")
        prompt = """
        I recently purchased your premium software subscription, but I'm having trouble accessing some of the 
        features that were advertised. The AI assistant and advanced analytics dashboard aren't showing up in 
        my account. I've tried logging out and back in, clearing my cache, and using different browsers, but 
        nothing seems to work. My subscription confirmation email shows I should have access to these features. 
        Can you help me resolve this issue?
        """
        
        # Load sample customer support dataset
        sample_data = [
            {"question": "How do I reset my password?", "expected_length": 100},
            {"question": "I was charged twice for my subscription. How can I get a refund?", "expected_length": 150},
            {"question": "The mobile app keeps crashing when I try to upload files. What should I do?", "expected_length": 180},
            {"question": "How do I cancel my subscription before the next billing cycle?", "expected_length": 130},
            {"question": "I need to change my shipping address for an order I just placed.", "expected_length": 120},
        ]
    
    # Create output directory
    os.makedirs("demo_results", exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run benchmarks with different configurations
    benchmark_results = {}
    
    # 1. Baseline model
    logger.info("Loading baseline model")
    baseline_model = load_model(model_name)
    benchmark_results["Baseline"] = run_benchmark(baseline_model, tokenizer, prompt)
    del baseline_model
    torch.cuda.empty_cache()
    
    # 2. Efficiency-Transformer only
    logger.info("Loading model with Efficiency-Transformer")
    efficiency_model = load_model(model_name, use_efficiency_transformer=True)
    benchmark_results["Efficiency-Transformer"] = run_benchmark(efficiency_model, tokenizer, prompt)
    del efficiency_model
    torch.cuda.empty_cache()
    
    # 3. Pruning only
    logger.info("Loading model with Pruning")
    pruned_model = load_model(model_name, use_pruning=True)
    benchmark_results["Pruning"] = run_benchmark(pruned_model, tokenizer, prompt)
    del pruned_model
    torch.cuda.empty_cache()
    
    # 4. Hybrid LoRA-Adapter only
    logger.info("Loading model with Hybrid LoRA-Adapter")
    hybrid_model = load_model(model_name, use_hybrid_adapter=True)
    benchmark_results["Hybrid LoRA-Adapter"] = run_benchmark(hybrid_model, tokenizer, prompt)
    del hybrid_model
    torch.cuda.empty_cache()
    
    # 5. All optimizations
    logger.info("Loading model with all Artemis optimizations")
    artemis_model = load_model(
        model_name, 
        use_efficiency_transformer=True,
        use_pruning=True,
        use_hybrid_adapter=True,
        quantize=True,
    )
    benchmark_results["Artemis (All optimizations)"] = run_benchmark(artemis_model, tokenizer, prompt)
    
    # Run detailed benchmark on Artemis model with all optimizations
    logger.info("Running detailed resource benchmark on Artemis model")
    detailed_results = run_detailed_resource_benchmark(artemis_model, tokenizer, prompt)
    
    # Generate outputs for sample questions
    logger.info("Generating outputs for sample questions with Artemis model")
    sample_outputs = []
    
    for item in tqdm(sample_data, desc="Generating outputs"):
        # Generate output
        inputs = tokenizer(item["question"], return_tensors="pt").to(artemis_model.device)
        
        start_time = time.time()
        with torch.no_grad():
            output_ids = artemis_model.generate(
                **inputs,
                max_new_tokens=item["expected_length"],
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generation_time = time.time() - start_time
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        # Add to results
        sample_outputs.append({
            "question": item["question"],
            "response": response,
            "tokens": len(tokenizer.encode(response)),
            "generation_time": generation_time,
            "tokens_per_second": len(tokenizer.encode(response)) / generation_time,
        })
    
    # Save all results
    results = {
        "domain": domain,
        "model_name": model_name,
        "benchmark_results": benchmark_results,
        "detailed_resource_benchmark": detailed_results,
        "sample_outputs": sample_outputs,
    }
    
    output_file = os.path.join("demo_results", f"{domain}_demo_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate comparison chart
    chart_path = os.path.join("demo_results", f"{domain}_benchmark_comparison.png")
    generate_comparison_chart(benchmark_results, chart_path)
    
    # Create a summary table
    summary = pd.DataFrame({
        "Model": list(benchmark_results.keys()),
        "Latency (s)": [results["avg_latency_seconds"] for results in benchmark_results.values()],
        "Memory (GB)": [results["avg_memory_gb"] for results in benchmark_results.values()],
        "Tokens/sec": [results["tokens_per_second"] for results in benchmark_results.values()],
    })
    
    # Calculate improvements compared to baseline
    baseline_latency = summary.loc[summary["Model"] == "Baseline", "Latency (s)"].values[0]
    baseline_memory = summary.loc[summary["Model"] == "Baseline", "Memory (GB)"].values[0]
    baseline_tokens_per_sec = summary.loc[summary["Model"] == "Baseline", "Tokens/sec"].values[0]
    
    summary["Latency Improvement"] = ((baseline_latency - summary["Latency (s)"]) / baseline_latency * 100).map("{:.1f}%".format)
    summary["Memory Improvement"] = ((baseline_memory - summary["Memory (GB)"]) / baseline_memory * 100).map("{:.1f}%".format)
    summary["Speed Improvement"] = ((summary["Tokens/sec"] - baseline_tokens_per_sec) / baseline_tokens_per_sec * 100).map("{:.1f}%".format)
    
    # Save and print summary table
    summary_path = os.path.join("demo_results", f"{domain}_summary.csv")
    summary.to_csv(summary_path, index=False)
    
    print("\n--- Performance Comparison Summary ---")
    print(summary)
    print(f"\nDetailed results saved to {output_file}")
    print(f"Comparison chart saved to {chart_path}")
    
    # Print sample output
    print("\n--- Sample Output ---")
    print(f"Question: {sample_outputs[0]['question']}")
    print(f"Response: {sample_outputs[0]['response']}")
    print(f"Generation speed: {sample_outputs[0]['tokens_per_second']:.2f} tokens/sec")
    
    return results


def main():
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description="Artemis Demonstration")
    parser.add_argument("--domain", type=str, default="legal", choices=["legal", "medical", "customer-support"],
                      help="Domain for the demonstration")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help="HuggingFace model name or path")
    
    args = parser.parse_args()
    
    # Run the demonstration
    run_real_world_example(domain=args.domain, model_name=args.model)


if __name__ == "__main__":
    main()
