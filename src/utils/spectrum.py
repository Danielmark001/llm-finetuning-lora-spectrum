"""
Spectrum Fine-Tuning Implementation
==================================
This module implements the Spectrum fine-tuning technique for LLMs,
which analyzes Signal-to-Noise Ratio (SNR) of model layers to efficiently
select which layers to fine-tune.

Based on the research: "Spectrum: Analyzing and Exploiting the Language Model's 
Frequency Domain for Parameter-Efficient Fine-Tuning"
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
from transformers import PreTrainedModel
import logging

logger = logging.getLogger(__name__)


class SpectrumAnalyzer:
    """
    Analyzes model layers using Signal-to-Noise Ratio (SNR) to determine
    which layers are most important for fine-tuning.
    """
    
    def __init__(self, model: PreTrainedModel):
        """
        Initialize the Spectrum analyzer with a pre-trained model.
        
        Args:
            model: The pre-trained language model to analyze
        """
        self.model = model
        self.layers_snr = {}
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """
        Detect the type of model architecture to determine layer naming convention.
        
        Returns:
            str: Model architecture type (e.g., 'llama', 'mistral', 'gemma', etc.)
        """
        # Extract model type from the model class name
        model_name = self.model.__class__.__name__.lower()
        
        if 'llama' in model_name:
            return 'llama'
        elif 'mistral' in model_name:
            return 'mistral'
        elif 'gemma' in model_name:
            return 'gemma'
        elif 'gpt' in model_name:
            return 'gpt'
        else:
            # Default to llama-type architecture
            logger.warning(f"Unknown model type: {model_name}. Defaulting to 'llama' architecture.")
            return 'llama'
    
    def analyze_layer_snr(self) -> Dict[str, float]:
        """
        Analyze the Signal-to-Noise Ratio (SNR) of each layer in the model.
        SNR is calculated by dividing the mean of absolute weight values by their standard deviation.
        
        Returns:
            Dict[str, float]: Mapping of layer names to their SNR values
        """
        model_name = self.model_type
        snr_values = {}
        
        # Get appropriate layer prefix based on model architecture
        if model_name == 'llama':
            layer_prefix = 'model.layers.'
        elif model_name == 'mistral':
            layer_prefix = 'model.layers.'
        elif model_name == 'gemma':
            layer_prefix = 'model.layers.'
        elif model_name == 'gpt':
            layer_prefix = 'transformer.h.'
        else:
            layer_prefix = 'model.layers.'
            
        # Iterate through model layers
        for name, param in self.model.named_parameters():
            if layer_prefix in name and param.requires_grad:
                # Extract layer number
                try:
                    layer_parts = name.split(layer_prefix)[1]
                    layer_num = int(layer_parts.split('.')[0])
                    
                    # Calculate SNR for this parameter
                    values = param.detach().cpu().numpy().flatten()
                    mean_abs = np.mean(np.abs(values))
                    std = np.std(values)
                    
                    # SNR = mean(|W|) / std(W)
                    snr = mean_abs / (std + 1e-10)  # Add epsilon to avoid division by zero
                    
                    # Store SNR by layer number
                    key = f"layer_{layer_num}"
                    if key not in snr_values:
                        snr_values[key] = []
                    snr_values[key].append(snr)
                except Exception as e:
                    logger.warning(f"Could not process layer: {name}. Error: {str(e)}")
        
        # Average SNR across parameters within each layer
        self.layers_snr = {layer: np.mean(snrs) for layer, snrs in snr_values.items()}
        
        return self.layers_snr
    
    def get_trainable_layers_by_snr(self, threshold: float = 0.5) -> List[int]:
        """
        Select layers to train based on SNR values and a threshold.
        
        Args:
            threshold: SNR percentile threshold (0.0 to 1.0) for selecting layers
                      Higher values = fewer layers selected
        
        Returns:
            List[int]: Layer indices to fine-tune
        """
        # Analyze layers if not done yet
        if not self.layers_snr:
            self.analyze_layer_snr()
            
        # Convert layer names to indices
        layer_indices = {int(name.split('_')[1]): snr for name, snr in self.layers_snr.items()}
        
        # Sort layers by SNR in descending order
        sorted_layers = sorted(layer_indices.items(), key=lambda x: x[1], reverse=True)
        
        # Select layers above the SNR threshold
        snr_values = [snr for _, snr in sorted_layers]
        snr_threshold_value = np.percentile(snr_values, threshold * 100)
        
        selected_layers = [layer_idx for layer_idx, snr in sorted_layers if snr >= snr_threshold_value]
        
        logger.info(f"Selected {len(selected_layers)} layers to train with SNR threshold {threshold}")
        return sorted(selected_layers)  # Sort numerically
    
    def freeze_layers_except(self, trainable_layers: List[int]) -> None:
        """
        Freeze all layers except those selected for fine-tuning.
        
        Args:
            trainable_layers: List of layer indices to keep trainable
        """
        model_name = self.model_type
        
        # Get appropriate layer prefix based on model architecture
        if model_name == 'llama':
            layer_prefix = 'model.layers.'
        elif model_name == 'mistral':
            layer_prefix = 'model.layers.'
        elif model_name == 'gemma':
            layer_prefix = 'model.layers.'
        elif model_name == 'gpt':
            layer_prefix = 'transformer.h.'
        else:
            layer_prefix = 'model.layers.'
        
        # Convert to set for faster lookups
        trainable_layers_set = set(trainable_layers)
        
        # Freeze/unfreeze layers
        for name, param in self.model.named_parameters():
            if layer_prefix in name:
                try:
                    # Extract layer number
                    layer_parts = name.split(layer_prefix)[1]
                    layer_num = int(layer_parts.split('.')[0])
                    
                    # Set requires_grad based on whether this layer should be trained
                    should_train = layer_num in trainable_layers_set
                    param.requires_grad = should_train
                    
                except Exception as e:
                    logger.warning(f"Could not process layer for freezing: {name}. Error: {str(e)}")
            
            # Always keep embeddings and output layer trainable
            elif "embed" in name.lower() or name.startswith("lm_head"):
                param.requires_grad = True
    
    def get_layer_distribution_stats(self) -> Dict[str, float]:
        """
        Calculate statistics about layer SNR distribution for reporting.
        
        Returns:
            Dict[str, float]: Statistics about SNR distribution
        """
        if not self.layers_snr:
            self.analyze_layer_snr()
            
        snr_values = list(self.layers_snr.values())
        
        stats = {
            "mean_snr": float(np.mean(snr_values)),
            "median_snr": float(np.median(snr_values)),
            "min_snr": float(np.min(snr_values)),
            "max_snr": float(np.max(snr_values)),
            "std_snr": float(np.std(snr_values)),
        }
        
        return stats
    
    def generate_layer_snr_report(self) -> str:
        """
        Generate a human-readable report of layer SNR values.
        
        Returns:
            str: Formatted report of layer SNR values
        """
        if not self.layers_snr:
            self.analyze_layer_snr()
            
        # Get layers sorted by SNR
        sorted_layers = sorted(self.layers_snr.items(), key=lambda x: int(x[0].split('_')[1]))
        
        # Generate report
        report = "Layer Signal-to-Noise Ratio (SNR) Analysis\n"
        report += "=" * 50 + "\n\n"
        
        for layer_name, snr in sorted_layers:
            layer_idx = int(layer_name.split('_')[1])
            report += f"Layer {layer_idx:3d}: {snr:.6f}\n"
        
        report += "\n" + "-" * 50 + "\n"
        
        # Add statistics
        stats = self.get_layer_distribution_stats()
        report += f"Mean SNR:   {stats['mean_snr']:.6f}\n"
        report += f"Median SNR: {stats['median_snr']:.6f}\n"
        report += f"Min SNR:    {stats['min_snr']:.6f}\n"
        report += f"Max SNR:    {stats['max_snr']:.6f}\n"
        report += f"Std Dev:    {stats['std_snr']:.6f}\n"
        
        return report


def get_optimal_layers_for_model(model: PreTrainedModel, threshold: float = 0.5) -> List[int]:
    """
    Convenience function to get optimal layers to train for a given model.
    
    Args:
        model: The pre-trained language model
        threshold: SNR percentile threshold (0.0 to 1.0)
    
    Returns:
        List[int]: Layer indices recommended for fine-tuning
    """
    analyzer = SpectrumAnalyzer(model)
    return analyzer.get_trainable_layers_by_snr(threshold)
