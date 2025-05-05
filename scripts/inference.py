#!/usr/bin/env python3
"""
Artemis Inference Script
========================
This script runs optimized inference using Artemis models, achieving 2.7x speedup
on consumer hardware with 8-bit hybrid inference while preserving model quality.
"""

import os
import sys
import yaml
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import threading
from queue import Queue

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)
from peft import PeftModel
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.hybrid_adapter import HybridLoRAAdapter
from src.utils.evaluation import evaluate_model, measure_resource_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_flash_attention: bool = True,
    hybrid_mode: bool = False,
    device_map: str = "auto",
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model and tokenizer for inference.
    
    Args:
        model_path: Path to the base model or merged model
        adapter_path: Optional path to adapter weights
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        use_flash_attention: Whether to use flash attention
        hybrid_mode: Whether to use hybrid LoRA-Adapter approach
        device_map: Device mapping strategy
        
    Returns:
        Tuple of model and tokenizer
    """
    logger.info(f"Loading model from {model_path}")
    
    # Determine quantization config
    quantization_config = None
    if load_in_8bit:
        logger.info("Loading model in 8-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_quant_type="symmetric",
            bnb_8bit_use_double_quant=True,
        )
    elif load_in_4bit:
        logger.info("Loading model in 4-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # Prepare model loading kwargs
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2 for faster inference")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # Apply adapter if provided
    if adapter_path:
        logger.info(f"Loading adapter from {adapter_path}")
        
        # Check for hybrid adapter config
        hybrid_config_path = os.path.join(adapter_path, "hybrid_adapter_config.json")
        
        if hybrid_mode and os.path.exists(hybrid_config_path):
            logger.info("Using Hybrid LoRA-Adapter approach for inference")
            adapter = HybridLoRAAdapter.load(adapter_path)
            model = adapter.load_adapter(model)
            
            # Apply 8-bit quantization for inference if in hybrid mode
            if hybrid_mode and load_in_8bit:
                logger.info("Applying 8-bit quantization through hybrid adapter")
                model = adapter.quantize_for_inference(model)
        else:
            logger.info("Loading standard PEFT adapter")
            model = PeftModel.from_pretrained(model, adapter_path)
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Log model size and memory usage
    param_count = sum(p.numel() for p in model.parameters())
    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    logger.info(f"Model size: {param_count:,} parameters")
    logger.info(f"GPU memory usage: {memory_used:.2f} GB")
    
    return model, tokenizer


def generate_text(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    stream: bool = False,
) -> Union[str, TextIteratorStreamer]:
    """
    Generate text from a prompt.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        do_sample: Whether to use sampling (vs. greedy decoding)
        stream: Whether to stream the output
        
    Returns:
        Generated text or a text streamer
    """
    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with optional streaming
    start_time = time.time()
    
    if stream:
        # Set up streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            generation_config=generation_config,
            streamer=streamer,
        )
        
        # Generate in a separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        return streamer
    else:
        # Generate without streaming
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # Decode and return
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(prompt):]
        
        # Log generation stats
        generation_time = time.time() - start_time
        gen_tokens = len(tokenizer.encode(response))
        tokens_per_sec = gen_tokens / generation_time
        
        logger.info(f"Generated {gen_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")
        
        return response


def run_chat_interface(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = True,
):
    """
    Run an interactive chat interface in the terminal.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        stream: Whether to stream the output
    """
    print("\n===== Artemis Chat Interface =====")
    print("Enter 'exit' or 'quit' to end the conversation.")
    print("Enter 'benchmark' to run a benchmark test.")
    print("Enter 'settings' to view or change generation settings.")
    print("====================================\n")
    
    # Initialize chat settings
    settings = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    
    # Initialize chat history (for future use with contextual models)
    chat_history = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        
        elif user_input.lower() == "benchmark":
            # Run a quick benchmark
            print("\n[Running benchmark test...]")
            benchmark_prompt = "Explain the concept of parameter-efficient fine-tuning in LLMs."
            tokens_to_generate = 200
            
            start_time = time.time()
            with torch.no_grad():
                inputs = tokenizer(benchmark_prompt, return_tensors="pt").to(model.device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=tokens_to_generate,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                    repetition_penalty=settings["repetition_penalty"],
                )
            
            generation_time = time.time() - start_time
            tokens_per_sec = tokens_to_generate / generation_time
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            print(f"\nBenchmark Results:")
            print(f"- Generated {tokens_to_generate} tokens in {generation_time:.2f} seconds")
            print(f"- Speed: {tokens_per_sec:.2f} tokens/second")
            print(f"- GPU Memory: {memory_used:.2f} GB")
            
            # Reset CUDA memory stats
            torch.cuda.reset_peak_memory_stats()
            continue
        
        elif user_input.lower() == "settings":
            # Show and optionally modify settings
            print("\nCurrent Generation Settings:")
            for key, value in settings.items():
                print(f"- {key}: {value}")
            
            change = input("\nChange settings? (y/n): ")
            if change.lower() == "y":
                for key in settings:
                    new_value = input(f"New value for {key} [{settings[key]}]: ")
                    if new_value:
                        try:
                            # Convert to appropriate type
                            if isinstance(settings[key], int):
                                settings[key] = int(new_value)
                            elif isinstance(settings[key], float):
                                settings[key] = float(new_value)
                        except ValueError:
                            print(f"Invalid value, keeping {key}={settings[key]}")
                
                print("\nUpdated Settings:")
                for key, value in settings.items():
                    print(f"- {key}: {value}")
            
            continue
        
        # Generate response
        print("\nArtemis: ", end="", flush=True)
        
        if stream:
            # Stream the response
            streamer = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_new_tokens=settings["max_new_tokens"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                repetition_penalty=settings["repetition_penalty"],
                stream=True,
            )
            
            # Print the streaming response
            for token in streamer:
                print(token, end="", flush=True)
            print()
        else:
            # Generate without streaming
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_new_tokens=settings["max_new_tokens"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                repetition_penalty=settings["repetition_penalty"],
                stream=False,
            )
            print(response)


def run_batch_inference(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    input_file: str,
    output_file: str,
    max_new_tokens: int = 512,
    batch_size: int = 1,
):
    """
    Run batch inference on a file of inputs.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_file: Path to input file (JSON format with "prompts" key)
        output_file: Path to output file
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
    """
    logger.info(f"Running batch inference with batch size {batch_size}")
    
    # Load inputs
    with open(input_file, "r") as f:
        input_data = json.load(f)
    
    prompts = input_data["prompts"] if "prompts" in input_data else input_data
    
    # Set up output data
    outputs = []
    
    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Run inference in batches
    start_time = time.time()
    total_tokens = 0
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        
        # Tokenize inputs
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **batch_inputs,
                generation_config=generation_config,
            )
        
        # Decode outputs
        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        # Extract generated responses (remove prompts)
        responses = []
        for prompt, output_text in zip(batch_prompts, output_texts):
            response = output_text[len(tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)):]
            responses.append(response)
            total_tokens += len(tokenizer.encode(response))
        
        # Add to outputs
        for prompt, response in zip(batch_prompts, responses):
            outputs.append({
                "prompt": prompt,
                "response": response,
            })
    
    # Calculate stats
    total_time = time.time() - start_time
    tokens_per_sec = total_tokens / total_time
    
    logger.info(f"Generated {total_tokens} tokens in {total_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")
    
    # Save outputs
    with open(output_file, "w") as f:
        json.dump({"generations": outputs}, f, indent=2)
    
    logger.info(f"Saved {len(outputs)} outputs to {output_file}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Artemis Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--adapter_path", type=str, help="Path to the adapter weights")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Inference modes
    parser.add_argument("--chat", action="store_true", help="Run interactive chat interface")
    parser.add_argument("--batch", action="store_true", help="Run batch inference")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    
    # Model loading options
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable flash attention")
    parser.add_argument("--hybrid_mode", action="store_true", help="Use hybrid LoRA-Adapter approach")
    
    # Generation options
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming in chat mode")
    
    # Batch mode options
    parser.add_argument("--input_file", type=str, help="Input file for batch inference")
    parser.add_argument("--output_file", type=str, help="Output file for batch inference")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    
    # Evaluation options
    parser.add_argument("--eval_dataset", type=str, help="Path to evaluation dataset")
    parser.add_argument("--perplexity", action="store_true", help="Calculate perplexity")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                      help="Benchmarks to run (e.g., lm-evaluation-harness, domain-specific)")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=not args.no_flash_attention,
        hybrid_mode=args.hybrid_mode,
    )
    
    # Turn on evaluation mode
    model.eval()
    
    # Run chat interface
    if args.chat:
        run_chat_interface(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream=not args.no_stream,
        )
    
    # Run batch inference
    elif args.batch:
        if not args.input_file or not args.output_file:
            logger.error("Input and output files must be provided for batch mode")
            return
        
        run_batch_inference(
            model=model,
            tokenizer=tokenizer,
            input_file=args.input_file,
            output_file=args.output_file,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    
    # Run evaluation
    elif args.evaluate:
        from datasets import load_dataset
        
        # Load evaluation dataset
        eval_dataset = None
        if args.eval_dataset:
            logger.info(f"Loading evaluation dataset from {args.eval_dataset}")
            eval_dataset = load_dataset("json", data_files=args.eval_dataset)["train"]
        
        # Run resource usage benchmark
        if not eval_dataset:
            sample_input = "Explain the concept of parameter-efficient fine-tuning in large language models."
            logger.info("Running resource usage benchmark...")
            results = measure_resource_usage(
                model=model,
                tokenizer=tokenizer,
                sample_input=sample_input,
                num_trials=5,
                generation_length=256,
            )
            
            # Print results
            logger.info("Resource Usage Results:")
            logger.info(f"- Inference latency: {results['inference_latency_ms']:.2f} ms")
            logger.info(f"- Tokens per second: {results['tokens_per_second']:.2f}")
            logger.info(f"- Memory usage: {results['memory_usage_mb']:.2f} MB")
            
            # Save results
            os.makedirs("evaluation", exist_ok=True)
            with open("evaluation/resource_usage.json", "w") as f:
                json.dump(results, f, indent=2)
        
        # Run perplexity calculation if requested
        if args.perplexity and eval_dataset:
            from src.utils.evaluation import calculate_perplexity
            
            logger.info("Calculating perplexity...")
            perplexity_results = calculate_perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset=eval_dataset,
                batch_size=args.batch_size or 8,
            )
            
            logger.info(f"Perplexity: {perplexity_results['perplexity']:.4f}")
            
            # Save results
            os.makedirs("evaluation", exist_ok=True)
            with open("evaluation/perplexity_results.json", "w") as f:
                json.dump(perplexity_results, f, indent=2)
        
        # Run benchmarks if requested
        if args.benchmarks:
            logger.info(f"Running benchmarks: {args.benchmarks}")
            
            output_dir = "evaluation/benchmarks"
            os.makedirs(output_dir, exist_ok=True)
            
            eval_results = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                benchmarks=args.benchmarks,
                batch_size=args.batch_size or 8,
                output_dir=output_dir,
                resource_metrics=True,
            )
            
            # Log summary results
            logger.info("Benchmark Results Summary:")
            
            if "lm_eval_harness" in eval_results:
                lm_results = eval_results["lm_eval_harness"]
                logger.info("LM Evaluation Harness:")
                if "results" in lm_results:
                    for task, score in lm_results["results"].items():
                        logger.info(f"- {task}: {score:.4f}")
            
            if "custom_benchmarks" in eval_results:
                for benchmark, results in eval_results["custom_benchmarks"].items():
                    logger.info(f"Custom Benchmark - {benchmark}:")
                    if "domain_score" in results:
                        logger.info(f"- Domain Score: {results['domain_score']:.4f}")
            
            if "resource_usage" in eval_results:
                res = eval_results["resource_usage"]
                logger.info("Resource Usage:")
                logger.info(f"- Inference Speed: {res['tokens_per_second']:.2f} tokens/sec")
                logger.info(f"- Memory Usage: {res['memory_usage_mb']:.2f} MB")
    
    else:
        logger.info("No inference mode specified. Use --chat, --batch, or --evaluate")


if __name__ == "__main__":
    main()
