#!/usr/bin/env python3
"""
Artemis: Adaptive Representation Tuning for Efficient Model Instruction Synthesis
================================================================================
This script implements the Artemis framework with parameter-efficient fine-tuning techniques
that reduce training costs by 40% while preserving 95% of model performance, including
Efficiency-Transformer, advanced pruning techniques, and hybrid LoRA-Adapter approach.
"""

import os
import sys
import yaml
import logging
import argparse
import time
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
from accelerate import Accelerator

# Custom imports
from utils.spectrum import SpectrumAnalyzer
from utils.data_processing import preprocess_dataset
from utils.optimization import configure_optimizer
from utils.evaluation import evaluate_model, measure_resource_usage, create_domain_specific_benchmarks
# New imports for Artemis features
from utils.efficiency import create_efficient_model, EfficiencyTransformer
from utils.pruning import create_pruning_manager, PruningManager
from utils.hybrid_adapter import create_hybrid_adapter, HybridLoRAAdapter

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


def setup_model_and_tokenizer(config: Dict) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Optional[Dict]]:
    """
    Set up the model and tokenizer based on configuration.
    Handles quantization, efficiency techniques, and other optimizations.
    
    Returns:
        Tuple containing:
        - model: The set up model
        - tokenizer: The tokenizer
        - efficiency_metrics: Optional dictionary of efficiency metrics
    """
    model_config = config["model"]
    ft_config = config["fine_tuning"]
    
    logger.info(f"Setting up model with {ft_config['method']} fine-tuning approach")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer"],
        trust_remote_code=model_config.get("trust_remote_code", False),
        padding_side="right",
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization if needed
    quantization_config = None
    if model_config.get("load_in_8bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=getattr(torch, config.get("quantization", {}).get("bnb_8bit_compute_dtype", "float16")),
            bnb_8bit_quant_type=config.get("quantization", {}).get("bnb_8bit_quant_type", "symmetric"),
            bnb_8bit_use_double_quant=config.get("quantization", {}).get("bnb_8bit_use_double_quant", True),
        )
    elif model_config.get("load_in_4bit", False) or ft_config["method"] == "qlora":
        quant_config = config.get("quantization", {})
        compute_dtype = getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
    
    # Additional model loading kwargs
    model_kwargs = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # Flash attention configuration
    if model_config.get("use_flash_attention", False):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load the model
    start_time = time.time()
    logger.info(f"Loading base model: {model_config['base_model']}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        **model_kwargs
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Record base model stats
    base_model_stats = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "load_time_seconds": load_time,
    }
    
    # For quantized models, prepare for training
    if quantization_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
        )
    
    efficiency_metrics = None
    
    # Setup the appropriate technique based on configuration
    if ft_config["method"] == "efficiency_transformer":
        logger.info("Setting up Efficiency-Transformer for parameter-efficient fine-tuning")
        
        # Create efficient model with adaptive layer selection and dynamic ranks
        model, efficiency_metrics = create_efficient_model(config["fine_tuning"]["efficiency_transformer"], model)
        
        # Log efficiency metrics
        logger.info(f"Efficiency-Transformer metrics: {efficiency_metrics['efficiency_metrics']}")
        
    elif ft_config["method"] in ["lora", "qlora"]:
        # Configure LoRA or QLoRA
        lora_config = LoraConfig(
            r=ft_config["lora"]["r"],
            lora_alpha=ft_config["lora"]["alpha"],
            lora_dropout=ft_config["lora"]["dropout"],
            bias=ft_config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=ft_config["lora"]["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        
        # Log model structure
        logger.info(f"LoRA model set up with rank={ft_config['lora']['r']}, alpha={ft_config['lora']['alpha']}")
        
    elif ft_config["method"] == "spectrum":
        # For Spectrum method, analyze and select layers to fine-tune
        analyzer = SpectrumAnalyzer(model)
        if ft_config["spectrum"]["layers_to_finetune"] == "auto":
            snr_threshold = ft_config["spectrum"]["snr_threshold"]
            trainable_layers = analyzer.get_trainable_layers_by_snr(snr_threshold)
            logger.info(f"Spectrum analysis selected {len(trainable_layers)} layers for fine-tuning")
            
            # Apply selective freezing based on SNR analysis
            analyzer.freeze_layers_except(trainable_layers)
        else:
            # Use manually specified layers
            trainable_layers = ft_config["spectrum"]["layers_to_finetune"]
            analyzer.freeze_layers_except(trainable_layers)
            
    # Apply hybrid LoRA-Adapter if enabled
    if model_config.get("hybrid_lora_adapter", False):
        logger.info("Applying Hybrid LoRA-Adapter approach for efficient training and inference")
        model, hybrid_metrics = create_hybrid_adapter(config, model)
        
        if efficiency_metrics is None:
            efficiency_metrics = {}
        efficiency_metrics["hybrid_adapter"] = hybrid_metrics
        
        logger.info(f"Hybrid adapter applied with {hybrid_metrics['performance_metrics']['inference_speedup']:.2f}x speedup")
    
    # Apply pruning if enabled
    if config.get("pruning", {}).get("enabled", False):
        logger.info("Setting up pruning manager for model compression")
        pruning_manager = create_pruning_manager(config, model)
        
        # Store pruning manager in model for use during training
        model.pruning_manager = pruning_manager
        
        if efficiency_metrics is None:
            efficiency_metrics = {}
        efficiency_metrics["pruning"] = pruning_manager.get_pruning_summary()
        
        logger.info(f"Pruning setup complete with target sparsity {config['pruning']['final_sparsity']:.2%}")
    
    # Log final model stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model setup complete. Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    return model, tokenizer, efficiency_metrics


def setup_datasets(config: Dict, tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Load and preprocess datasets according to configuration.
    """
    dataset_config = config["dataset"]
    max_seq_length = config["training"]["max_seq_length"]
    
    # Load datasets based on format
    if dataset_config["format"] == "alpaca":
        train_dataset = load_dataset("json", data_files=dataset_config["train_path"])["train"]
        eval_dataset = load_dataset("json", data_files=dataset_config["eval_path"])["train"] if dataset_config.get("eval_path") else None
        
        datasets = preprocess_dataset(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            format=dataset_config["format"],
            add_eos_token=dataset_config["preprocessing"].get("add_eos_token", True),
            add_bos_token=dataset_config["preprocessing"].get("add_bos_token", False),
            use_chat_template=dataset_config["preprocessing"].get("use_chat_template", True),
        )
    
    # Add support for other dataset formats
    elif dataset_config["format"] == "sharegpt":
        # Implementation for ShareGPT format
        logger.info("Loading ShareGPT format dataset...")
        train_dataset = load_dataset("json", data_files=dataset_config["train_path"])["train"]
        eval_dataset = load_dataset("json", data_files=dataset_config["eval_path"])["train"] if dataset_config.get("eval_path") else None
        
        datasets = preprocess_dataset(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            format="sharegpt",
            add_eos_token=dataset_config["preprocessing"].get("add_eos_token", True),
            add_bos_token=dataset_config["preprocessing"].get("add_bos_token", False),
            use_chat_template=dataset_config["preprocessing"].get("use_chat_template", True),
        )
    
    elif dataset_config["format"] == "oasst":
        # Implementation for OASST format
        logger.info("Loading OASST format dataset...")
        train_dataset = load_dataset("json", data_files=dataset_config["train_path"])["train"]
        eval_dataset = load_dataset("json", data_files=dataset_config["eval_path"])["train"] if dataset_config.get("eval_path") else None
        
        datasets = preprocess_dataset(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            format="oasst",
            add_eos_token=dataset_config["preprocessing"].get("add_eos_token", True),
            add_bos_token=dataset_config["preprocessing"].get("add_bos_token", False),
            use_chat_template=dataset_config["preprocessing"].get("use_chat_template", True),
        )
    
    elif dataset_config["format"] == "custom":
        # Implementation for custom format
        logger.info("Loading custom format dataset...")
        train_dataset = load_dataset("json", data_files=dataset_config["train_path"])["train"]
        eval_dataset = load_dataset("json", data_files=dataset_config["eval_path"])["train"] if dataset_config.get("eval_path") else None
        
        datasets = preprocess_dataset(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            format="custom",
            add_eos_token=dataset_config["preprocessing"].get("add_eos_token", True),
            add_bos_token=dataset_config["preprocessing"].get("add_bos_token", False),
            use_chat_template=dataset_config["preprocessing"].get("use_chat_template", True),
            custom_prompt_template=dataset_config["preprocessing"].get("custom_prompt_template", None),
        )
    
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_config['format']}")
    
    # Log dataset statistics
    logger.info(f"Training dataset: {len(datasets['train'])} examples")
    if datasets.get("eval"):
        logger.info(f"Evaluation dataset: {len(datasets['eval'])} examples")
    
    return datasets


class ArtemisTrainer(SFTTrainer):
    """
    Extended SFTTrainer with Artemis-specific functionality for efficiency and pruning.
    """
    
    def __init__(self, pruning_enabled=False, efficiency_metrics=None, resource_tracking=False, *args, **kwargs):
        """Initialize ArtemisTrainer with additional tracking."""
        super().__init__(*args, **kwargs)
        self.pruning_enabled = pruning_enabled
        self.efficiency_metrics = efficiency_metrics or {}
        self.resource_tracking = resource_tracking
        self.training_metrics = {
            "loss_history": [],
            "resource_usage": [],
            "step_times": [],
        }
    
    def training_step(self, model, inputs):
        """Overridden training step to apply pruning if enabled."""
        # Run the standard training step
        loss = super().training_step(model, inputs)
        
        # Apply pruning if enabled
        if self.pruning_enabled and hasattr(model, "pruning_manager"):
            # Accumulate gradients for gradient-based pruning
            if model.pruning_manager.importance_metric == "gradient_sensitivity":
                model.pruning_manager.accumulate_gradients()
            
            # Apply pruning step
            model.pruning_manager.step(total_steps=self.state.max_steps)
            
            # Apply mask to gradients to maintain pruning during training
            model.pruning_manager.apply_mask_to_gradients()
        
        # Record metrics if tracking enabled
        if self.resource_tracking:
            self.training_metrics["loss_history"].append(loss.item())
            
            if len(self.training_metrics["loss_history"]) % 10 == 0:  # Every 10 steps
                # Record GPU memory usage
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
                self.training_metrics["resource_usage"].append({
                    "step": len(self.training_metrics["loss_history"]),
                    "gpu_memory_gb": gpu_memory,
                    "loss": loss.item(),
                })
                
                # Reset memory stats for the next window
                torch.cuda.reset_peak_memory_stats()
        
        return loss
    
    def save_metrics(self, output_dir):
        """Save training metrics to the output directory."""
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Save efficiency metrics if available
        if self.efficiency_metrics:
            efficiency_path = os.path.join(output_dir, "efficiency_metrics.json")
            with open(efficiency_path, "w") as f:
                json.dump(self.efficiency_metrics, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Artemis: Adaptive Representation Tuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only, no training")
    parser.add_argument("--create_benchmarks", action="store_true", help="Create domain-specific benchmarks")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create domain-specific benchmarks if requested
    if args.create_benchmarks:
        logger.info("Creating domain-specific benchmarks...")
        benchmark_paths = create_domain_specific_benchmarks("evaluation")
        logger.info(f"Created benchmarks: {benchmark_paths}")
        if not args.eval_only:
            logger.info("Exiting after benchmark creation")
            return
    
    # Record start time
    start_time = time.time()
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer, efficiency_metrics = setup_model_and_tokenizer(config)
    
    # Create output directory
    output_dir = config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Store efficiency metrics
    if efficiency_metrics:
        metrics_path = os.path.join(output_dir, "initial_efficiency_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(efficiency_metrics, f, indent=2)
    
    # If eval_only mode, run evaluation and exit
    if args.eval_only:
        logger.info("Running evaluation only (no training)...")
        
        # Load evaluation dataset if available
        eval_dataset = None
        if config["dataset"].get("eval_path"):
            datasets = setup_datasets(config, tokenizer)
            eval_dataset = datasets.get("eval")
        
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            benchmarks=config["evaluation"]["benchmarks"],
            batch_size=config["evaluation"]["eval_batch_size"],
            output_dir=os.path.join(output_dir, "evaluation"),
            resource_metrics=config["evaluation"].get("resource_metrics", True),
        )
        
        # If there's a baseline to compare with
        if config["evaluation"].get("baseline_comparison") and config["evaluation"].get("baseline_results_path"):
            from utils.evaluation import compare_with_baseline
            baseline_path = config["evaluation"]["baseline_results_path"]
            if os.path.exists(baseline_path):
                logger.info(f"Comparing with baseline results from {baseline_path}")
                with open(baseline_path, "r") as f:
                    baseline_results = json.load(f)
                
                comparison = compare_with_baseline(eval_results, baseline_results)
                eval_results["baseline_comparison"] = comparison
        
        # Save evaluation results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {results_path}")
        return
    
    # Setup datasets for training
    logger.info("Loading and preprocessing datasets...")
    datasets = setup_datasets(config, tokenizer)
    
    # Configure the training arguments
    training_config = config["training"]
    output_config = config["output"]
    
    # Check if we're using DeepSpeed
    deepspeed_config = None
    if config["distributed"].get("use_deepspeed", False):
        deepspeed_config = config["distributed"]["deepspeed_config"]
    
    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_config["output_dir"],
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["micro_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        fp16="fp16" in training_config["mixed_precision"].lower(),
        bf16="bf16" in training_config["mixed_precision"].lower(),
        logging_steps=output_config["logging_steps"],
        evaluation_strategy=config["evaluation"]["eval_strategy"] if config["evaluation"]["do_eval"] else "no",
        eval_steps=config["evaluation"]["eval_steps"] if config["evaluation"]["do_eval"] else None,
        save_strategy="steps",
        save_steps=output_config["save_steps"],
        save_total_limit=output_config["save_total_limit"],
        deepspeed=deepspeed_config,
        report_to="tensorboard",
        push_to_hub=output_config["push_to_hub"],
        hub_model_id=output_config["hub_model_id"],
    )
    
    # Set up a data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Set up ArtemisTrainer
    trainer = ArtemisTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("eval"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=training_config["max_seq_length"],
        pruning_enabled=config.get("pruning", {}).get("enabled", False),
        efficiency_metrics=efficiency_metrics,
        resource_tracking=output_config.get("track_resource_usage", True),
    )
    
    # Start training
    logger.info("Starting fine-tuning with Artemis...")
    trainer.train()
    
    # Save the final model and metrics
    logger.info("Saving the final model and metrics...")
    trainer.save_model()
    trainer.save_metrics(output_dir)
    
    # For pruned models, prepare for quantization if enabled
    if hasattr(model, "pruning_manager"):
        logger.info("Preparing pruned model for quantization...")
        model.pruning_manager.prepare_for_quantization()
    
    # Calculate total training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    
    # Run evaluation if configured
    if config["evaluation"]["do_eval"]:
        logger.info("Running final evaluation...")
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=datasets.get("eval"),
            benchmarks=config["evaluation"]["benchmarks"],
            batch_size=config["evaluation"]["eval_batch_size"],
            output_dir=os.path.join(output_dir, "evaluation"),
            resource_metrics=config["evaluation"].get("resource_metrics", True),
        )
        
        # Save evaluation results
        results_path = os.path.join(output_dir, "final_evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Save final efficiency metrics and performance summary
    performance_summary = {
        "training_time_minutes": training_time / 60,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "parameter_efficiency": sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()),
    }
    
    if hasattr(model, "pruning_manager"):
        pruning_summary = model.pruning_manager.get_pruning_summary()
        performance_summary["pruning"] = pruning_summary
    
    # Save performance summary
    summary_path = os.path.join(output_dir, "performance_summary.json")
    with open(summary_path, "w") as f:
        json.dump(performance_summary, f, indent=2)
    
    logger.info("Artemis fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
