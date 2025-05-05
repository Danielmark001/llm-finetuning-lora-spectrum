#!/usr/bin/env python
"""
Tool for converting models to different formats.

This script provides utilities to convert Artemis models between different
formats including:
- Converting from PyTorch to ONNX format
- Converting to TensorRT for NVIDIA GPUs
- Converting to CoreML for Apple devices
- Converting to TensorFlow Lite
- Exporting for different quantization levels (INT8, INT4, etc.)
"""

import argparse
import logging
import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Add parent directory to path to allow importing from artemis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artemis.utils.hybrid_adapter import HybridAdapterConfig
from artemis.utils.quantization import quantize_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Artemis Model Conversion Tool")
    
    # Input model specification
    parser.add_argument(
        "--input_model", 
        type=str,
        required=True,
        help="Path to input model directory or checkpoint"
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
    
    # Output specification
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="Name for the output model files (defaults to original model name)"
    )
    
    # Conversion options
    parser.add_argument(
        "--target_format",
        type=str,
        required=True,
        choices=[
            "onnx", "tensorrt", "coreml", "tflite", 
            "openvino", "torchscript", "pytorch"
        ],
        help="Target format for conversion"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "dynamic", "static", "int8", "int4", "float16"],
        default="none",
        help="Quantization method to apply"
    )
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        help="Path to calibration dataset for static quantization"
    )
    
    # Optimization options
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply additional optimizations specific to target format"
    )
    parser.add_argument(
        "--keep_adapters",
        action="store_true",
        help="Keep LoRA/Adapter weights separate (if supported by format)"
    )
    parser.add_argument(
        "--merge_adapters",
        action="store_true",
        help="Merge LoRA/Adapter weights into base model"
    )
    
    # Additional options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for exported model"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for exported model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for conversion (cuda or cpu)"
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """
    Load model and tokenizer based on model type and configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple of (model, tokenizer, config)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load model based on type
    logger.info(f"Loading {args.model_type} model from {args.input_model}")
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(args.input_model)
        
        # Load model with appropriate settings
        if args.device == "cuda" and torch.cuda.is_available():
            if args.quantization in ["int8", "int4"]:
                logger.info(f"Loading with {args.quantization} quantization")
                
                # Set appropriate loading parameters based on quantization
                load_in_8bit = args.quantization == "int8"
                load_in_4bit = args.quantization == "int4"
                
                model = AutoModelForCausalLM.from_pretrained(
                    args.input_model,
                    device_map="auto",
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.input_model,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.input_model,
                trust_remote_code=True,
            )
        
        logger.info(f"Successfully loaded model and tokenizer")
        
        # If config was not provided but model has it, extract it
        if config is None and hasattr(model, "config"):
            config = model.config.to_dict()
        
        return model, tokenizer, config
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def apply_optimizations(model, args, config):
    """
    Apply optimizations to the model before conversion.
    
    Args:
        model: The model to optimize
        args: Command line arguments
        config: Model configuration
        
    Returns:
        Optimized model
    """
    # Apply quantization if requested
    if args.quantization != "none" and not any(q in args.quantization for q in ["int8", "int4"]):
        logger.info(f"Applying {args.quantization} quantization")
        model = quantize_model(model, method=args.quantization)
    
    # Apply format-specific optimizations
    if args.optimize:
        logger.info(f"Applying optimizations for {args.target_format}")
        
        if args.target_format == "tensorrt":
            # TensorRT specific optimizations
            from artemis.utils.optimization import optimize_for_tensorrt
            model = optimize_for_tensorrt(model)
            
        elif args.target_format == "onnx":
            # ONNX specific optimizations
            from artemis.utils.optimization import optimize_for_onnx
            model = optimize_for_onnx(model)
            
        elif args.target_format == "openvino":
            # OpenVINO specific optimizations
            from artemis.utils.optimization import optimize_for_openvino
            model = optimize_for_openvino(model)
    
    # Handle adapter merging if requested
    if args.merge_adapters and hasattr(model, "merge_adapter_weights"):
        logger.info("Merging adapter weights into base model")
        model = model.merge_adapter_weights()
    
    return model

def convert_to_onnx(model, tokenizer, args):
    """
    Convert model to ONNX format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        args: Command line arguments
        
    Returns:
        Path to converted model
    """
    logger.info("Converting model to ONNX format")
    
    output_path = Path(args.output_dir) / f"{args.output_name}.onnx"
    
    # Prepare example inputs
    dummy_input = tokenizer(
        "This is a sample input for ONNX conversion", 
        return_tensors="pt"
    ).to(model.device)
    
    # Define dynamic axes for variable sequence length
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'outputs': {0: 'batch_size', 1: 'sequence_length'}
    }
    
    # Export the model
    torch.onnx.export(
        model,
        (dummy_input.input_ids, dummy_input.attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['outputs'],
        dynamic_axes=dynamic_axes,
        opset_version=12,
        do_constant_folding=True,
        export_params=True,
    )
    
    logger.info(f"ONNX model saved to {output_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    logger.info("ONNX model verification passed")
    
    return output_path

def convert_to_tensorrt(model, tokenizer, args):
    """
    Convert model to TensorRT format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        args: Command line arguments
        
    Returns:
        Path to converted model
    """
    logger.info("Converting model to TensorRT format")
    
    # First convert to ONNX
    onnx_path = convert_to_onnx(model, tokenizer, args)
    
    # Then convert ONNX to TensorRT
    from artemis.utils.tensorrt import convert_onnx_to_tensorrt
    
    output_path = Path(args.output_dir) / f"{args.output_name}.engine"
    
    convert_onnx_to_tensorrt(
        onnx_path, 
        output_path,
        fp16=args.quantization == "float16",
        int8=args.quantization == "int8",
        max_batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length
    )
    
    logger.info(f"TensorRT model saved to {output_path}")
    
    return output_path

def convert_to_coreml(model, tokenizer, args):
    """
    Convert model to CoreML format for Apple devices.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        args: Command line arguments
        
    Returns:
        Path to converted model
    """
    logger.info("Converting model to CoreML format")
    
    try:
        import coremltools as ct
    except ImportError:
        logger.error("coremltools not installed. Please install with: pip install coremltools")
        sys.exit(1)
    
    output_path = Path(args.output_dir) / f"{args.output_name}.mlmodel"
    
    # Prepare example inputs
    dummy_input = tokenizer(
        "This is a sample input for CoreML conversion", 
        return_tensors="pt"
    ).to('cpu')  # CoreML conversion needs CPU tensors
    
    # Convert to TorchScript first
    torchscript_model = torch.jit.trace(
        model.to('cpu'), 
        (dummy_input.input_ids, dummy_input.attention_mask)
    )
    
    # Convert to CoreML
    mlmodel = ct.convert(
        torchscript_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=dummy_input.input_ids.shape),
            ct.TensorType(name="attention_mask", shape=dummy_input.attention_mask.shape)
        ],
        compute_units="ALL"  # Use CPU, GPU, and Neural Engine
    )
    
    # Save the model
    mlmodel.save(output_path)
    
    logger.info(f"CoreML model saved to {output_path}")
    
    return output_path

def convert_to_tflite(model, tokenizer, args):
    """
    Convert model to TensorFlow Lite format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        args: Command line arguments
        
    Returns:
        Path to converted model
    """
    logger.info("Converting model to TensorFlow Lite format")
    
    # First convert to ONNX
    onnx_path = convert_to_onnx(model, tokenizer, args)
    
    # Then convert ONNX to TFLite
    try:
        import onnx
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        logger.error("Required packages not installed. Please install with: pip install onnx tf2onnx tensorflow")
        sys.exit(1)
    
    output_path = Path(args.output_dir) / f"{args.output_name}.tflite"
    
    # Convert ONNX to TensorFlow format
    onnx_model = onnx.load(onnx_path)
    tf_rep = tf2onnx.convert.from_onnx(onnx_model)
    tf_model = tf_rep.tensorflow_model
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
    
    # Set optimization flags
    if args.quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # If calibration dataset is provided, use it for quantization
        if args.calibration_dataset:
            def representative_dataset():
                # Load and process calibration dataset
                # This is a simplified example
                for i in range(100):  # Use 100 samples for calibration
                    yield [tf.random.normal([1, args.max_sequence_length])]
            
            converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logger.info(f"TensorFlow Lite model saved to {output_path}")
    
    return output_path

def convert_to_openvino(model, tokenizer, args):
    """
    Convert model to OpenVINO format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        args: Command line arguments
        
    Returns:
        Path to converted model
    """
    logger.info("Converting model to OpenVINO format")
    
    # First convert to ONNX
    onnx_path = convert_to_onnx(model, tokenizer, args)
    
    try:
        from openvino.runtime import Core
        from openvino.tools.mo import convert_model
    except ImportError:
        logger.error("OpenVINO not installed. Please install with: pip install openvino")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.output_name}.xml"
    
    # Convert ONNX to OpenVINO IR
    ov_model = convert_model(
        onnx_path,
        compress_to_fp16=(args.quantization == "float16"),
        output_dir=str(output_dir),
        model_name=args.output_name
    )
    
    logger.info(f"OpenVINO model saved to {output_path}")
    
    return output_path

def convert_to_torchscript(model, tokenizer, args):
    """
    Convert model to TorchScript format.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        args: Command line arguments
        
    Returns:
        Path to converted model
    """
    logger.info("Converting model to TorchScript format")
    
    output_path = Path(args.output_dir) / f"{args.output_name}.pt"
    
    # Prepare example inputs
    dummy_input = tokenizer(
        "This is a sample input for TorchScript conversion", 
        return_tensors="pt"
    ).to(model.device)
    
    # Convert to TorchScript using tracing
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (dummy_input.input_ids, dummy_input.attention_mask)
        )
        
        # Save the model
        torch.jit.save(traced_model, output_path)
    
    logger.info(f"TorchScript model saved to {output_path}")
    
    return output_path

def save_pytorch_model(model, tokenizer, args):
    """
    Save model in PyTorch format (with optional quantization).
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        args: Command line arguments
        
    Returns:
        Path to saved model
    """
    logger.info("Saving model in PyTorch format")
    
    output_dir = Path(args.output_dir)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    with open(output_dir / "conversion_info.json", 'w') as f:
        import json
        json.dump({
            "original_model": args.input_model,
            "model_type": args.model_type,
            "quantization": args.quantization,
            "optimized": args.optimize,
            "adapters_merged": args.merge_adapters,
            "conversion_date": str(Path.ctime(Path.ctime)),
        }, f, indent=2)
    
    logger.info(f"PyTorch model saved to {output_dir}")
    
    return output_dir

def main():
    """Main function to convert models."""
    args = parse_arguments()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default output name if not provided
    if not args.output_name:
        args.output_name = Path(args.input_model).stem
    
    try:
        # Load model and tokenizer
        model, tokenizer, config = load_model_and_tokenizer(args)
        
        # Apply optimizations
        model = apply_optimizations(model, args, config)
        
        # Convert to target format
        if args.target_format == "onnx":
            convert_to_onnx(model, tokenizer, args)
        elif args.target_format == "tensorrt":
            convert_to_tensorrt(model, tokenizer, args)
        elif args.target_format == "coreml":
            convert_to_coreml(model, tokenizer, args)
        elif args.target_format == "tflite":
            convert_to_tflite(model, tokenizer, args)
        elif args.target_format == "openvino":
            convert_to_openvino(model, tokenizer, args)
        elif args.target_format == "torchscript":
            convert_to_torchscript(model, tokenizer, args)
        elif args.target_format == "pytorch":
            save_pytorch_model(model, tokenizer, args)
        else:
            logger.error(f"Unsupported target format: {args.target_format}")
            sys.exit(1)
        
        logger.info("Model conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model conversion: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
