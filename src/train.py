#!/usr/bin/env python3
"""
Advanced LLM Fine-Tuning Script
================================
This script implements state-of-the-art methods for fine-tuning large language models
using parameter-efficient techniques such as LoRA, QLoRA, and Spectrum.
"""

import os
import sys
import yaml
import logging
import argparse
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
from utils.evaluation import evaluate_model

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


def setup_model_and_tokenizer(config: Dict) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Set up the model and tokenizer based on configuration.
    Handles quantization, flash attention, and other optimizations.
    """
    model_config = config["model"]
    ft_config = config["fine_tuning"]
    
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
    if ft_config["method"] == "qlora":
        quant_config = ft_config["quantization"]
        compute_dtype = getattr(torch, quant_config["bnb_4bit_compute_dtype"])
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
    elif model_config.get("load_in_8bit", False):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif model_config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
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
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        **model_kwargs
    )
    
    # For quantized models, prepare for training
    if quantization_config and ft_config["method"] in ["lora", "qlora"]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
        )

    # Configure LoRA or QLoRA
    if ft_config["method"] in ["lora", "qlora"]:
        lora_config = LoraConfig(
            r=ft_config["lora"]["r"],
            lora_alpha=ft_config["lora"]["alpha"],
            lora_dropout=ft_config["lora"]["dropout"],
            bias=ft_config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=ft_config["lora"]["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        
    # For Spectrum method, analyze and select layers to fine-tune
    elif ft_config["method"] == "spectrum":
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
    
    return model, tokenizer


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
        pass
    
    elif dataset_config["format"] == "oasst":
        # Implementation for OASST format
        pass
    
    elif dataset_config["format"] == "custom":
        # Implementation for custom format
        pass
    
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_config['format']}")
    
    return datasets


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Advanced LLM Fine-Tuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup datasets
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
    
    # Set up SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("eval"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=training_config["max_seq_length"],
    )
    
    # Start training
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving the final model...")
    trainer.save_model()
    
    # Run evaluation if configured
    if config["evaluation"]["do_eval"]:
        logger.info("Running final evaluation...")
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=datasets.get("eval"),
            benchmarks=config["evaluation"]["benchmarks"],
            batch_size=config["evaluation"]["eval_batch_size"],
        )
        
        # Log the results
        for benchmark, results in eval_results.items():
            logger.info(f"{benchmark} results: {results}")
    
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
