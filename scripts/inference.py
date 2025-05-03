#!/usr/bin/env python3
"""
LLM Fine-Tuning Inference Script
================================
This script provides inference capabilities for fine-tuned language models,
including chat interfaces, batch inference, and model evaluation.
"""

import os
import sys
import yaml
import json
import logging
import argparse
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    GenerationConfig,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import project utilities
from src.utils.evaluation import evaluate_model, calculate_perplexity

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


def load_model_for_inference(
    model_path: str,
    config: Dict = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_flash_attention: bool = False,
    adapter_path: Optional[str] = None,
    device_map: str = "auto",
) -> tuple:
    """
    Load a model for inference with optimizations.
    
    Args:
        model_path: Path to the model or model identifier
        config: Optional configuration dictionary
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        use_flash_attention: Whether to use flash attention
        adapter_path: Path to the PEFT adapter
        device_map: Device mapping strategy
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}...")
    
    # Override config with function arguments if provided
    if config:
        if load_in_8bit is False and config["model"].get("load_in_8bit", False):
            load_in_8bit = True
        if load_in_4bit is False and config["model"].get("load_in_4bit", False):
            load_in_4bit = True
        if use_flash_attention is False and config["model"].get("use_flash_attention", False):
            use_flash_attention = True
    
    # Quantization config
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": device_map,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # Flash attention configuration
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # Load adapter if provided
    if adapter_path:
        logger.info(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    logger.info(f"Model loaded successfully with {model.num_parameters():,} parameters")
    return model, tokenizer


def interactive_chat(
    model,
    tokenizer,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Run an interactive chat session with the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        system_prompt: Optional system prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    """
    # Check if tokenizer supports chat template
    has_chat_template = hasattr(tokenizer, "apply_chat_template")
    
    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    
    # Initialize conversation history
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    
    print("\n\033[1mInteractive Chat Session\033[0m")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to clear the conversation history.")
    print("Type 'save' to save the conversation history.")
    print("----------------------------------------")
    
    while True:
        # Get user input
        user_input = input("\n\033[94mYou: \033[0m")
        
        # Handle special commands
        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting chat session. Goodbye!")
            break
        
        elif user_input.lower() == "clear":
            if system_prompt:
                history = [{"role": "system", "content": system_prompt}]
            else:
                history = []
            print("\nConversation history cleared.")
            continue
        
        elif user_input.lower().startswith("save"):
            # Extract filename if provided
            parts = user_input.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "conversation.json"
            
            # Save conversation
            with open(filename, "w") as f:
                json.dump(history, f, indent=2)
            
            print(f"\nConversation saved to {filename}")
            continue
        
        # Add user message to history
        history.append({"role": "user", "content": user_input})
        
        # Format prompt based on tokenizer capabilities
        if has_chat_template:
            prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback to a simple prompt format
            prompt = ""
            for message in history:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            prompt += "Assistant: "
        
        # Generate response
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        output = model.generate(
            input_ids,
            generation_config=generation_config,
        )
        
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the assistant's response
        if has_chat_template:
            # For chat template, we need to extract the last assistant message
            assistant_response = generated_text.split("Assistant: ")[-1].strip()
        else:
            # For simple format, take everything after the last "Assistant: "
            assistant_response = generated_text.split("Assistant: ")[-1].strip()
        
        # Add assistant message to history
        history.append({"role": "assistant", "content": assistant_response})
        
        # Print the response
        print(f"\n\033[92mAssistant: \033[0m{assistant_response}")


def batch_inference(
    model,
    tokenizer,
    input_file: str,
    output_file: str,
    prompt_field: str = "prompt",
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
):
    """
    Run batch inference on a dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_file: Path to the input JSON file
        output_file: Path to save the output
        prompt_field: Field name containing the prompts
        batch_size: Batch size for inference
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        system_prompt: Optional system prompt
    """
    logger.info(f"Running batch inference on {input_file}...")
    
    # Load the input data
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    
    # Create a pipeline for text generation
    pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        generation_config=generation_config,
    )
    
    # Check if tokenizer supports chat template
    has_chat_template = hasattr(tokenizer, "apply_chat_template")
    
    # Process each input
    results = []
    total = len(data)
    
    for i in range(0, total, batch_size):
        batch = data[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        
        batch_prompts = []
        for item in batch:
            prompt = item[prompt_field]
            
            # Add system prompt if provided
            if system_prompt and has_chat_template:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif system_prompt:
                prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant: "
            
            batch_prompts.append(prompt)
        
        # Generate outputs
        outputs = pipeline(batch_prompts)
        
        # Process outputs
        for j, output in enumerate(outputs):
            item_result = batch[j].copy()
            
            # Get the generated text
            if isinstance(output, list):
                generated_text = output[0]["generated_text"]
            else:
                generated_text = output["generated_text"]
            
            # Extract model response (after the prompt)
            item_result["full_output"] = generated_text
            
            # Try to extract just the model's response
            try:
                if has_chat_template:
                    response = generated_text.split("Assistant: ")[-1].strip()
                else:
                    response = generated_text.split(batch_prompts[j])[-1].strip()
                
                item_result["model_response"] = response
            except Exception as e:
                logger.warning(f"Error extracting response: {e}")
                item_result["model_response"] = generated_text
            
            results.append(item_result)
    
    # Save the results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Batch inference complete. Results saved to {output_file}")
    return results


def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Inference")
    
    # Model loading arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or model identifier")
    parser.add_argument("--adapter_path", type=str, help="Path to the PEFT adapter")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use flash attention")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--chat", action="store_true", help="Run interactive chat mode")
    mode_group.add_argument("--batch", action="store_true", help="Run batch inference mode")
    mode_group.add_argument("--evaluate", action="store_true", help="Run model evaluation")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--system_prompt", type=str, help="System prompt for chat and batch modes")
    
    # Batch mode arguments
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file for batch mode")
    parser.add_argument("--output_file", type=str, help="Path to save the output for batch mode")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="Field name containing the prompts")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    
    # Evaluation arguments
    parser.add_argument("--eval_dataset", type=str, help="Path to the evaluation dataset")
    parser.add_argument("--perplexity", action="store_true", help="Calculate perplexity")
    parser.add_argument("--benchmarks", type=str, nargs="+", help="Benchmarks to run")
    parser.add_argument("--eval_output_dir", type=str, help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load the model and tokenizer
    model, tokenizer = load_model_for_inference(
        model_path=args.model_path,
        config=config,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=args.use_flash_attention,
        adapter_path=args.adapter_path,
    )
    
    # Run the specified mode
    if args.chat:
        interactive_chat(
            model=model,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    
    elif args.batch:
        if not args.input_file or not args.output_file:
            parser.error("--input_file and --output_file required for batch mode")
        
        batch_inference(
            model=model,
            tokenizer=tokenizer,
            input_file=args.input_file,
            output_file=args.output_file,
            prompt_field=args.prompt_field,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
        )
    
    elif args.evaluate:
        if not args.eval_dataset:
            parser.error("--eval_dataset required for evaluation mode")
        
        from datasets import load_dataset
        
        # Load evaluation dataset
        if args.eval_dataset.endswith(".json"):
            eval_dataset = load_dataset("json", data_files=args.eval_dataset, split="train")
        elif args.eval_dataset.endswith(".csv"):
            eval_dataset = load_dataset("csv", data_files=args.eval_dataset, split="train")
        else:
            eval_dataset = load_dataset(args.eval_dataset, split="train")
        
        # Run evaluation
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            benchmarks=args.benchmarks,
            batch_size=args.batch_size,
            output_dir=args.eval_output_dir,
        )
        
        # Print results summary
        print("\nEvaluation Results:")
        print("===================")
        
        if "perplexity" in results:
            print(f"Perplexity: {results['perplexity']['perplexity']:.4f}")
        
        if "lm_eval_harness" in results:
            print("\nLM Evaluation Harness:")
            for task, metrics in results["lm_eval_harness"]["results"].items():
                print(f"  {task}: {metrics}")
        
        if "domain_specific" in results:
            print("\nDomain-Specific Evaluation:")
            for domain, metrics in results["domain_specific"].items():
                print(f"  {domain}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value}")
        
        # Save full results
        if args.eval_output_dir:
            os.makedirs(args.eval_output_dir, exist_ok=True)
            with open(os.path.join(args.eval_output_dir, "full_results.json"), "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
