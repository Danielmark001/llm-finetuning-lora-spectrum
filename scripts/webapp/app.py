#!/usr/bin/env python3
"""
LLM Fine-Tuning Web UI
======================
A Gradio web interface for managing LLM fine-tuning jobs.
"""

import os
import sys
import json
import yaml
import time
import logging
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
webapp_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(webapp_dir)
sys.path.insert(0, str(project_root))

try:
    import gradio as gr
    import pandas as pd
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import HfApi, list_models
    from src.utils.data_processing import calculate_dataset_statistics
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages: pip install gradio pandas numpy torch transformers huggingface_hub")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(script_dir, "webapp.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Global variables
RUNNING_JOBS = {}  # Track running jobs
JOB_LOGS = {}  # Store job logs
AVAILABLE_GPUS = []  # List of available GPUs
MODEL_CACHE = {}  # Cache for model info
DATASET_CACHE = {}  # Cache for dataset info


def get_available_gpus() -> List[int]:
    """Get list of available GPUs."""
    try:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return []


def refresh_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs."""
    global AVAILABLE_GPUS
    
    result = {
        "count": 0,
        "devices": [],
        "total_memory": 0,
        "free_memory": 0,
    }
    
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            result["count"] = count
            AVAILABLE_GPUS = list(range(count))
            
            for i in range(count):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "free_memory": round((torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024**3), 2),
                }
                result["devices"].append(device_info)
                result["total_memory"] += device_info["total_memory"]
                result["free_memory"] += device_info["free_memory"]
    
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
    
    return result


def list_huggingface_models(filter_string: str = "", model_type: str = "llm") -> List[Dict[str, str]]:
    """List available models from Hugging Face."""
    try:
        # Use cached results if available and not filtering
        cache_key = f"{model_type}_{filter_string}"
        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key]
        
        # List popular base models for fine-tuning
        base_models = [
            {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama-3.1-8B-Instruct"},
            {"id": "meta-llama/Llama-3.1-70B-Instruct", "name": "Llama-3.1-70B-Instruct"},
            {"id": "mistralai/Mistral-7B-Instruct-v0.2", "name": "Mistral-7B-Instruct-v0.2"},
            {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3-mini-4k-instruct"},
            {"id": "google/gemma-7b-it", "name": "Gemma-7B-Instruct"},
            {"id": "HuggingFaceH4/zephyr-7b-beta", "name": "Zephyr-7B-Beta"},
            {"id": "stabilityai/stablelm-zephyr-3b", "name": "StableLM-Zephyr-3B"},
            {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama-1.1B-Chat"},
        ]
        
        # Filter models based on filter string
        if filter_string:
            filter_lower = filter_string.lower()
            filtered_models = [
                model for model in base_models
                if filter_lower in model["id"].lower() or filter_lower in model["name"].lower()
            ]
        else:
            filtered_models = base_models
        
        # Cache results
        MODEL_CACHE[cache_key] = filtered_models
        
        return filtered_models
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []


def list_project_datasets() -> List[Dict[str, Any]]:
    """List datasets available in the project."""
    data_dir = os.path.join(project_root, "data")
    result = []
    
    try:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if file.endswith(".json") and os.path.isfile(file_path):
                    # Get file size
                    size_bytes = os.path.getsize(file_path)
                    size_mb = round(size_bytes / (1024 * 1024), 2)
                    
                    # Get format (alpaca or sharegpt)
                    format_type = "unknown"
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if data and isinstance(data, list):
                                if all(isinstance(item, dict) for item in data):
                                    # Check for alpaca format
                                    if "instruction" in data[0] and "output" in data[0]:
                                        format_type = "alpaca"
                                    # Check for ShareGPT format
                                    elif "conversations" in data[0]:
                                        format_type = "sharegpt"
                    except Exception as e:
                        logger.warning(f"Error determining format for {file}: {e}")
                    
                    # Get example count
                    example_count = 0
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            example_count = len(data) if isinstance(data, list) else 0
                    except Exception:
                        pass
                    
                    result.append({
                        "name": file,
                        "path": file_path,
                        "size": size_mb,
                        "format": format_type,
                        "examples": example_count,
                    })
    
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
    
    return sorted(result, key=lambda x: x["name"])


def analyze_dataset(dataset_path: str) -> Dict[str, Any]:
    """Analyze a dataset and return statistics."""
    if dataset_path in DATASET_CACHE:
        return DATASET_CACHE[dataset_path]
    
    result = {
        "error": None,
        "format": "unknown",
        "examples": 0,
        "statistics": {},
    }
    
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
            
            if not isinstance(data, list):
                result["error"] = "Dataset must be a JSON array"
                return result
            
            result["examples"] = len(data)
            
            if not data:
                result["error"] = "Dataset is empty"
                return result
            
            # Detect format
            first_item = data[0]
            if "instruction" in first_item and "output" in first_item:
                result["format"] = "alpaca"
                
                # Calculate basic statistics
                instruction_lengths = [len(item.get("instruction", "").split()) for item in data]
                input_lengths = [len(item.get("input", "").split()) for item in data]
                output_lengths = [len(item.get("output", "").split()) for item in data]
                
                # Count examples with inputs
                has_input = sum(1 for item in data if item.get("input", "").strip())
                
                result["statistics"] = {
                    "examples": len(data),
                    "with_input": has_input,
                    "without_input": len(data) - has_input,
                    "avg_instruction_length": round(sum(instruction_lengths) / len(data), 1),
                    "avg_input_length": round(sum(input_lengths) / len(data), 1) if has_input else 0,
                    "avg_output_length": round(sum(output_lengths) / len(data), 1),
                    "min_output_length": min(output_lengths),
                    "max_output_length": max(output_lengths),
                }
                
            elif "conversations" in first_item:
                result["format"] = "sharegpt"
                
                # Calculate basic statistics
                conversation_lengths = [len(item.get("conversations", [])) for item in data]
                
                # Count roles
                role_counts = {}
                total_messages = 0
                
                for item in data:
                    for msg in item.get("conversations", []):
                        role = msg.get("role", "").lower()
                        role_counts[role] = role_counts.get(role, 0) + 1
                        total_messages += 1
                
                result["statistics"] = {
                    "examples": len(data),
                    "total_messages": total_messages,
                    "avg_conversation_length": round(sum(conversation_lengths) / len(data), 1),
                    "roles": role_counts,
                }
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error analyzing dataset {dataset_path}: {e}")
    
    # Cache results
    DATASET_CACHE[dataset_path] = result
    
    return result


def create_job_config(
    job_name: str,
    model_name: str,
    dataset_path: str,
    eval_path: Optional[str],
    method: str,
    output_dir: str,
    learning_rate: float,
    epochs: int,
    micro_batch_size: int,
    max_seq_length: int,
    gradient_accumulation_steps: int,
) -> Dict:
    """
    Create a configuration for a fine-tuning job.
    """
    # Determine dataset format
    dataset_info = analyze_dataset(dataset_path)
    dataset_format = dataset_info["format"]
    
    # Determine evaluation dataset format if provided
    eval_format = None
    if eval_path:
        eval_info = analyze_dataset(eval_path)
        eval_format = eval_info["format"]
    
    # Set appropriate target modules based on model name
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if "falcon" in model_name.lower():
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "mistral" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "llama" in model_name.lower() or "gemma" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Create configuration
    config = {
        "model": {
            "base_model": model_name,
            "tokenizer": model_name,
            "load_in_8bit": method == "lora",
            "load_in_4bit": method == "qlora",
            "trust_remote_code": True,
            "use_flash_attention": True,
        },
        "fine_tuning": {
            "method": method,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": target_modules,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "spectrum": {
                "snr_threshold": 0.5,
                "layers_to_finetune": "auto",
            },
            "quantization": {
                "bits": 4,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        },
        "training": {
            "epochs": epochs,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.3,
            "optimizer": "paged_adamw_8bit",
            "weight_decay": 0.001,
            "max_seq_length": max_seq_length,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
        },
        "dataset": {
            "format": dataset_format,
            "train_path": dataset_path,
            "eval_path": eval_path,
            "preprocessing": {
                "add_eos_token": True,
                "add_bos_token": False,
                "use_chat_template": True,
            },
        },
        "output": {
            "output_dir": output_dir,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 100,
            "save_total_limit": 5,
            "push_to_hub": False,
            "hub_model_id": None,
        },
        "evaluation": {
            "do_eval": eval_path is not None,
            "eval_batch_size": 8,
            "eval_strategy": "steps",
            "eval_steps": 100,
        },
    }
    
    # Get number of GPUs
    num_gpus = len(AVAILABLE_GPUS)
    
    # Add distributed training configuration if multiple GPUs
    if num_gpus > 1:
        # Create DeepSpeed config directory if it doesn't exist
        os.makedirs(os.path.join(project_root, "config"), exist_ok=True)
        ds_config_path = os.path.join(project_root, "config", f"ds_config_{job_name}.json")
        
        # Create a simple DeepSpeed config
        ds_config = {
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": 0.3,
            "zero_allow_untested_optimizer": True,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            }
        }
        
        # Save DeepSpeed config
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)
        
        # Add distributed config to main config
        config["distributed"] = {
            "use_deepspeed": True,
            "deepspeed_config": ds_config_path,
            "zero_stage": 2,
        }
    
    return config


def start_training_job(
    job_name: str,
    model_name: str,
    dataset_path: str,
    eval_path: Optional[str],
    method: str,
    output_dir: str,
    learning_rate: float,
    epochs: int,
    micro_batch_size: int,
    max_seq_length: int,
    gradient_accumulation_steps: int,
) -> Tuple[bool, str]:
    """Start a training job with the given parameters."""
    global RUNNING_JOBS, JOB_LOGS
    
    # Check if job already exists
    if job_name in RUNNING_JOBS:
        return False, f"Job '{job_name}' is already running"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create job configuration
    config = create_job_config(
        job_name=job_name,
        model_name=model_name,
        dataset_path=dataset_path,
        eval_path=eval_path,
        method=method,
        output_dir=output_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        micro_batch_size=micro_batch_size,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    # Save configuration
    config_path = os.path.join(output_dir, f"{job_name}_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    # Initialize log for the job
    JOB_LOGS[job_name] = []
    
    # Create log function
    def log_output(process):
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            log_line = line.strip()
            JOB_LOGS[job_name].append(log_line)
            logger.info(f"[{job_name}] {log_line}")
    
    # Determine command to run
    train_script = os.path.join(project_root, "src", "train.py")
    
    # Check if we need to use distributed training
    num_gpus = len(AVAILABLE_GPUS)
    
    if num_gpus > 1 and config.get("distributed", {}).get("use_deepspeed", False):
        # Use DeepSpeed for multi-GPU training
        command = [
            "deepspeed",
            f"--num_gpus={num_gpus}",
            train_script,
            "--config", config_path,
        ]
    else:
        # Use single GPU training
        command = [
            "python",
            train_script,
            "--config", config_path,
        ]
    
    try:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Store process information
        RUNNING_JOBS[job_name] = {
            "process": process,
            "command": " ".join(command),
            "config": config,
            "start_time": time.time(),
            "output_dir": output_dir,
        }
        
        # Start a thread to read output
        output_thread = threading.Thread(target=log_output, args=(process,))
        output_thread.daemon = True
        output_thread.start()
        
        # Start a thread to monitor the process
        monitor_thread = threading.Thread(target=monitor_job, args=(job_name,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return True, f"Job '{job_name}' started successfully"
    
    except Exception as e:
        logger.error(f"Error starting job '{job_name}': {e}")
        return False, f"Error starting job: {e}"


def monitor_job(job_name: str) -> None:
    """Monitor a running job and update its status."""
    global RUNNING_JOBS
    
    if job_name not in RUNNING_JOBS:
        return
    
    job_info = RUNNING_JOBS[job_name]
    process = job_info["process"]
    
    # Wait for the process to complete
    process.wait()
    
    # Update job information
    job_info["end_time"] = time.time()
    job_info["duration"] = job_info["end_time"] - job_info["start_time"]
    job_info["return_code"] = process.returncode
    
    if process.returncode == 0:
        job_info["status"] = "completed"
        logger.info(f"Job '{job_name}' completed successfully")
    else:
        job_info["status"] = "failed"
        logger.error(f"Job '{job_name}' failed with return code {process.returncode}")


def stop_job(job_name: str) -> Tuple[bool, str]:
    """Stop a running job."""
    global RUNNING_JOBS
    
    if job_name not in RUNNING_JOBS:
        return False, f"Job '{job_name}' not found"
    
    job_info = RUNNING_JOBS[job_name]
    process = job_info["process"]
    
    try:
        # Try to terminate the process
        process.terminate()
        
        # Wait for the process to terminate
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # If the process doesn't terminate, kill it
            process.kill()
        
        # Update job information
        job_info["end_time"] = time.time()
        job_info["duration"] = job_info["end_time"] - job_info["start_time"]
        job_info["status"] = "stopped"
        
        return True, f"Job '{job_name}' stopped successfully"
    
    except Exception as e:
        logger.error(f"Error stopping job '{job_name}': {e}")
        return False, f"Error stopping job: {e}"


def get_job_status(job_name: str) -> Dict[str, Any]:
    """Get status information for a job."""
    global RUNNING_JOBS, JOB_LOGS
    
    if job_name not in RUNNING_JOBS:
        return {"error": f"Job '{job_name}' not found"}
    
    job_info = RUNNING_JOBS[job_name]
    process = job_info["process"]
    
    # Check if the process is still running
    if process.poll() is None:
        status = "running"
    elif process.returncode == 0:
        status = "completed"
    else:
        status = "failed"
    
    # Get logs
    logs = JOB_LOGS.get(job_name, [])
    
    # Calculate duration
    start_time = job_info["start_time"]
    if "end_time" in job_info:
        end_time = job_info["end_time"]
    else:
        end_time = time.time()
    
    duration = end_time - start_time
    
    return {
        "name": job_name,
        "status": status,
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time if "end_time" in job_info else None,
        "return_code": process.returncode if process.poll() is not None else None,
        "logs": logs[-100:],  # Last 100 log lines
    }


def get_all_jobs() -> List[Dict[str, Any]]:
    """Get status information for all jobs."""
    global RUNNING_JOBS
    
    result = []
    
    for job_name in RUNNING_JOBS:
        job_info = RUNNING_JOBS[job_name]
        process = job_info["process"]
        
        # Check if the process is still running
        if process.poll() is None:
            status = "running"
        elif process.returncode == 0:
            status = "completed"
        else:
            status = "failed"
        
        # Calculate duration
        start_time = job_info["start_time"]
        if "end_time" in job_info:
            end_time = job_info["end_time"]
        else:
            end_time = time.time()
        
        duration = end_time - start_time
        
        result.append({
            "name": job_name,
            "status": status,
            "duration": duration,
            "model": job_info["config"]["model"]["base_model"],
            "method": job_info["config"]["fine_tuning"]["method"],
            "output_dir": job_info["output_dir"],
        })
    
    return result


def test_fine_tuned_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
) -> Tuple[bool, Union[str, Dict[str, Any]]]:
    """Load and test a fine-tuned model."""
    try:
        # Set up quantization config
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
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Load adapter if provided
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # Get model information
        model_info = {
            "name": model_path.split("/")[-1],
            "parameters": model.num_parameters(),
            "adapter": adapter_path.split("/")[-1] if adapter_path else None,
        }
        
        return True, model_info
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False, f"Error loading model: {e}"


# Create the Gradio app
def create_app():
    # Theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    )
    
    # Initialize app
    app = gr.Blocks(title="LLM Fine-Tuning UI", theme=theme)
    
    # Get initial GPU information
    gpu_info = refresh_gpu_info()
    
    with app:
        gr.Markdown("# LLM Fine-Tuning UI")
        gr.Markdown("A user interface for managing LLM fine-tuning jobs")
        
        with gr.Tabs():
            # New Job tab
            with gr.Tab("New Fine-tuning Job"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### System Information")
                        
                        gpu_count_info = gr.Textbox(
                            label="GPU Count",
                            value=f"{gpu_info['count']} GPUs detected" if gpu_info['count'] > 0 else "No GPUs detected",
                            interactive=False,
                        )
                        
                        gpu_memory_info = gr.Textbox(
                            label="Total GPU Memory",
                            value=f"{gpu_info['total_memory']:.2f} GB" if gpu_info['total_memory'] > 0 else "N/A",
                            interactive=False,
                        )
                        
                        refresh_button = gr.Button("Refresh System Info")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Job Configuration")
                        
                        job_name = gr.Textbox(
                            label="Job Name",
                            placeholder="Enter a name for this fine-tuning job",
                            value="fine-tuning-job-" + time.strftime("%Y%m%d-%H%M%S"),
                        )
                        
                        model_search = gr.Textbox(
                            label="Search Models",
                            placeholder="Search for models (e.g., 'llama', 'mistral')",
                        )
                        
                        model_dropdown = gr.Dropdown(
                            label="Select Base Model",
                            choices=[model["id"] for model in list_huggingface_models()],
                            value="meta-llama/Llama-3.1-8B-Instruct" if "meta-llama/Llama-3.1-8B-Instruct" in [model["id"] for model in list_huggingface_models()] else None,
                        )
                        
                        dataset_dropdown = gr.Dropdown(
                            label="Select Dataset",
                            choices=[dataset["path"] for dataset in list_project_datasets()],
                            value=None,
                        )
                        
                        eval_dataset_dropdown = gr.Dropdown(
                            label="Select Evaluation Dataset (Optional)",
                            choices=[dataset["path"] for dataset in list_project_datasets()],
                            value=None,
                        )
                        
                        dataset_info = gr.JSON(
                            label="Dataset Information",
                            value=None,
                        )
                        
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Method")
                        
                        method_dropdown = gr.Dropdown(
                            label="Fine-tuning Method",
                            choices=["qlora", "lora", "spectrum", "full"],
                            value="qlora",
                        )
                        
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            placeholder="Path to save fine-tuned model",
                            value=os.path.join(project_root, "models", "runs", "run-" + time.strftime("%Y%m%d-%H%M%S")),
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Parameters")
                        
                        learning_rate = gr.Slider(
                            label="Learning Rate",
                            minimum=1e-6,
                            maximum=1e-3,
                            value=2e-4,
                            step=1e-6,
                        )
                        
                        epochs = gr.Slider(
                            label="Epochs",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                        )
                        
                        micro_batch_size = gr.Slider(
                            label="Micro Batch Size",
                            minimum=1,
                            maximum=8,
                            value=1,
                            step=1,
                        )
                        
                        gradient_accumulation_steps = gr.Slider(
                            label="Gradient Accumulation Steps",
                            minimum=1,
                            maximum=64,
                            value=16,
                            step=1,
                        )
                        
                        max_seq_length = gr.Slider(
                            label="Max Sequence Length",
                            minimum=512,
                            maximum=8192,
                            value=4096,
                            step=512,
                        )
                
                start_button = gr.Button("Start Fine-tuning", variant="primary")
                job_status = gr.Markdown("")
            
            # Jobs tab
            with gr.Tab("Running Jobs"):
                gr.Markdown("### Active Fine-tuning Jobs")
                
                refresh_jobs_button = gr.Button("Refresh Jobs")
                
                jobs_table = gr.DataFrame(
                    label="Running Jobs",
                    headers=["Name", "Status", "Model", "Method", "Duration (min)", "Output Directory"],
                    datatype=["str", "str", "str", "str", "number", "str"],
                    row_count=10,
                )
                
                selected_job = gr.Textbox(label="Selected Job", visible=False)
                
                with gr.Row():
                    view_button = gr.Button("View Job Details")
                    stop_button = gr.Button("Stop Job", variant="stop")
                
                job_details = gr.JSON(
                    label="Job Details",
                    visible=False,
                )
                
                job_logs = gr.Textbox(
                    label="Job Logs",
                    lines=20,
                    visible=False,
                )
            
            # Models tab
            with gr.Tab("Fine-tuned Models"):
                gr.Markdown("### Test Fine-tuned Models")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        test_model_path = gr.Textbox(
                            label="Model Path",
                            placeholder="Path to fine-tuned model",
                        )
                        
                        test_adapter_path = gr.Textbox(
                            label="Adapter Path (Optional)",
                            placeholder="Path to adapter (for LoRA/QLoRA)",
                        )
                        
                        use_4bit_test = gr.Checkbox(
                            label="Use 4-bit Quantization",
                            value=True,
                        )
                        
                        use_8bit_test = gr.Checkbox(
                            label="Use 8-bit Quantization",
                            value=False,
                        )
                        
                        test_model_button = gr.Button("Test Model")
                        
                    with gr.Column(scale=1):
                        test_result = gr.JSON(
                            label="Model Information",
                        )
                        
                        prompt_input = gr.Textbox(
                            label="Test Prompt",
                            placeholder="Enter a prompt to test the model",
                            lines=3,
                        )
                        
                        generation_length = gr.Slider(
                            label="Maximum Generation Length",
                            minimum=16,
                            maximum=2048,
                            value=256,
                            step=16,
                        )
                        
                        generate_button = gr.Button("Generate")
                        
                        model_output = gr.Textbox(
                            label="Model Output",
                            lines=10,
                        )
        
        # Define event handlers
        def update_gpu_info():
            info = refresh_gpu_info()
            return (
                f"{info['count']} GPUs detected" if info['count'] > 0 else "No GPUs detected",
                f"{info['total_memory']:.2f} GB" if info['total_memory'] > 0 else "N/A",
            )
        
        def update_model_dropdown(search_text):
            models = list_huggingface_models(search_text)
            return gr.Dropdown(choices=[model["id"] for model in models])
        
        def update_dataset_info(dataset_path):
            if not dataset_path:
                return None
            
            info = analyze_dataset(dataset_path)
            return info
        
        def update_jobs_table():
            jobs = get_all_jobs()
            
            # Format jobs for the table
            table_data = []
            for job in jobs:
                table_data.append([
                    job["name"],
                    job["status"],
                    job["model"].split("/")[-1],
                    job["method"],
                    round(job["duration"] / 60, 2),
                    job["output_dir"],
                ])
            
            return table_data
        
        def start_job(
            job_name, model_name, dataset_path, eval_path, method, output_dir,
            learning_rate, epochs, micro_batch_size, max_seq_length, gradient_accumulation_steps
        ):
            if not job_name or not model_name or not dataset_path:
                return "### ❌ Error: Job name, model, and dataset are required"
            
            success, message = start_training_job(
                job_name=job_name,
                model_name=model_name,
                dataset_path=dataset_path,
                eval_path=eval_path,
                method=method,
                output_dir=output_dir,
                learning_rate=learning_rate,
                epochs=epochs,
                micro_batch_size=micro_batch_size,
                max_seq_length=max_seq_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            
            if success:
                return f"### ✅ {message}"
            else:
                return f"### ❌ {message}"
        
        def select_job_from_table(evt: gr.SelectData):
            jobs = get_all_jobs()
            selected_job_name = jobs[evt.index[0]]["name"]
            return selected_job_name
        
        def view_job_details(job_name):
            if not job_name:
                return gr.JSON.update(visible=False), gr.Textbox.update(visible=False)
            
            job_status = get_job_status(job_name)
            
            # Format for display
            details = {
                "name": job_status.get("name", ""),
                "status": job_status.get("status", ""),
                "duration": f"{job_status.get('duration', 0) / 60:.2f} minutes",
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job_status.get("start_time", 0))),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job_status.get("end_time", 0))) if job_status.get("end_time") else "N/A",
                "return_code": job_status.get("return_code", "N/A"),
            }
            
            logs = "\n".join(job_status.get("logs", []))
            
            return (
                gr.JSON.update(value=details, visible=True),
                gr.Textbox.update(value=logs, visible=True),
            )
        
        def stop_selected_job(job_name):
            if not job_name:
                return "No job selected"
            
            success, message = stop_job(job_name)
            if success:
                return message
            else:
                return message
        
        def test_model(model_path, adapter_path, use_4bit, use_8bit):
            if not model_path:
                return "Please enter a model path"
            
            # Don't allow both 4-bit and 8-bit
            if use_4bit and use_8bit:
                return "Cannot use both 4-bit and 8-bit quantization"
            
            success, result = test_fine_tuned_model(
                model_path=model_path,
                adapter_path=adapter_path if adapter_path else None,
                load_in_8bit=use_8bit,
                load_in_4bit=use_4bit,
            )
            
            if success:
                return result
            else:
                return {"error": result}
        
        def generate_text(model_path, adapter_path, use_4bit, use_8bit, prompt, max_length):
            if not model_path:
                return "Please enter a model path"
            
            if not prompt:
                return "Please enter a prompt"
            
            try:
                # Set up quantization config
                quantization_config = None
                if use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                elif use_8bit:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                
                # Load the tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                
                # Ensure the tokenizer has padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Model loading kwargs
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                }
                
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
                
                # Load the model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                
                # Load adapter if provided
                if adapter_path:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, adapter_path)
                
                # Format the prompt using chat template if available
                if hasattr(tokenizer, "apply_chat_template"):
                    messages = [
                        {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    formatted_prompt = prompt
                
                # Generate text
                input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                
                # Decode the output
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract the assistant's response
                if hasattr(tokenizer, "apply_chat_template"):
                    # Try to extract just the assistant's response
                    split_text = output_text.split("Assistant: ")
                    if len(split_text) > 1:
                        assistant_response = split_text[-1].strip()
                    else:
                        assistant_response = output_text
                else:
                    # Just return everything after the prompt
                    assistant_response = output_text[len(formatted_prompt):].strip()
                
                return assistant_response
            
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                return f"Error generating text: {e}"
        
        # Connect event handlers
        refresh_button.click(update_gpu_info, [], [gpu_count_info, gpu_memory_info])
        
        model_search.change(update_model_dropdown, [model_search], [model_dropdown])
        
        dataset_dropdown.change(update_dataset_info, [dataset_dropdown], [dataset_info])
        
        start_button.click(
            start_job,
            [
                job_name, model_dropdown, dataset_dropdown, eval_dataset_dropdown,
                method_dropdown, output_dir, learning_rate, epochs,
                micro_batch_size, max_seq_length, gradient_accumulation_steps
            ],
            [job_status],
        )
        
        refresh_jobs_button.click(update_jobs_table, [], [jobs_table])
        
        jobs_table.select(select_job_from_table, None, [selected_job])
        
        view_button.click(view_job_details, [selected_job], [job_details, job_logs])
        
        stop_button.click(stop_selected_job, [selected_job], [job_status])
        
        test_model_button.click(
            test_model,
            [test_model_path, test_adapter_path, use_4bit_test, use_8bit_test],
            [test_result],
        )
        
        generate_button.click(
            generate_text,
            [test_model_path, test_adapter_path, use_4bit_test, use_8bit_test, prompt_input, generation_length],
            [model_output],
        )
        
        # Initialize with active jobs
        gr.on_load(update_jobs_table, None, [jobs_table])
        
        # Update datasets dropdown on load
        gr.on_load(
            lambda: [dataset["path"] for dataset in list_project_datasets()],
            None,
            [dataset_dropdown, eval_dataset_dropdown],
        )
    
    return app


# Main function
def main():
    # Initialize
    global AVAILABLE_GPUS
    AVAILABLE_GPUS = get_available_gpus()
    
    # Create the app
    app = create_app()
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
    )


if __name__ == "__main__":
    main()
