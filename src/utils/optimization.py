"""
Optimization Utilities for LLM Fine-Tuning
==========================================
This module provides optimization tools such as custom optimizers, learning
rate schedulers, and gradient clipping strategies for LLM fine-tuning.
"""

import math
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_scheduler,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def configure_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    use_8bit_optimizer: bool = False,
) -> Optimizer:
    """
    Configure an optimizer based on the provided configuration.
    
    Args:
        model: The model to optimize
        config: Optimizer configuration dictionary
        use_8bit_optimizer: Whether to use bitsandbytes 8-bit optimizer
        
    Returns:
        Optimizer: Configured optimizer
    """
    # Extract optimization parameters
    lr = config.get("learning_rate", 2e-5)
    weight_decay = config.get("weight_decay", 0.01)
    optimizer_type = config.get("optimizer", "adamw").lower()
    
    # Get parameters to optimize
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer based on type
    if optimizer_type == "adamw":
        if use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    optimizer_grouped_parameters,
                    lr=lr,
                    betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                    eps=config.get("adam_epsilon", 1e-8),
                )
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to regular AdamW")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=lr,
                    betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                    eps=config.get("adam_epsilon", 1e-8),
                )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                eps=config.get("adam_epsilon", 1e-8),
            )
    
    elif optimizer_type == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=lr,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
            logger.info("Using Adafactor optimizer")
        except ImportError:
            logger.warning("Adafactor not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                eps=config.get("adam_epsilon", 1e-8),
            )
    
    elif optimizer_type == "paged_adamw_8bit" or optimizer_type == "paged_adamw_32bit":
        try:
            import bitsandbytes as bnb
            
            # Choose the correct optimizer
            if optimizer_type == "paged_adamw_8bit":
                optimizer = bnb.optim.PagedAdamW8bit(
                    optimizer_grouped_parameters,
                    lr=lr,
                    betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                    eps=config.get("adam_epsilon", 1e-8),
                )
                logger.info("Using 8-bit PagedAdamW optimizer")
            else:
                optimizer = bnb.optim.PagedAdamW32bit(
                    optimizer_grouped_parameters,
                    lr=lr,
                    betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                    eps=config.get("adam_epsilon", 1e-8),
                )
                logger.info("Using 32-bit PagedAdamW optimizer")
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to regular AdamW")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                eps=config.get("adam_epsilon", 1e-8),
            )
    
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            momentum=config.get("sgd_momentum", 0.9),
        )
        logger.info("Using SGD optimizer")
    
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
        logger.info("Using Adam optimizer")
    
    else:
        logger.warning(f"Unknown optimizer: {optimizer_type}, falling back to AdamW")
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
    
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to wrap
        scheduler_type: Type of scheduler (linear, cosine, constant, etc.)
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps (overrides warmup_ratio)
        warmup_ratio: Ratio of warmup steps (if num_warmup_steps not provided)
        last_epoch: Last epoch to resume from
        
    Returns:
        LambdaLR: Configured learning rate scheduler
    """
    # Calculate warmup steps
    if num_warmup_steps is None and warmup_ratio is not None:
        num_warmup_steps = int(num_training_steps * warmup_ratio)
    elif num_warmup_steps is None:
        num_warmup_steps = 0
    
    # Create scheduler based on type
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            last_epoch=last_epoch,
        )
    
    elif scheduler_type == "cosine_with_restarts":
        # Custom cosine with restarts implementation
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Get cycle length and number of cycles
            num_cycles = 1
            if "num_cycles" in scheduler_type:
                num_cycles = int(scheduler_type.split("_")[-1])
            
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return 0.0
            
            progress = progress * num_cycles
            cycle = math.floor(progress)
            progress = progress - cycle
            
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    
    else:
        # Use Transformers' built-in scheduler
        scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    return scheduler


class GradientClippingStrategy:
    """
    Gradient clipping strategy for model fine-tuning.
    """
    
    @staticmethod
    def clip_grad_norm(
        parameters: List[torch.nn.Parameter],
        max_grad_norm: float = 1.0,
        norm_type: float = 2.0,
    ) -> float:
        """
        Clip gradients by norm.
        
        Args:
            parameters: Model parameters
            max_grad_norm: Maximum gradient norm
            norm_type: Type of norm to use
            
        Returns:
            float: Gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, norm_type)
    
    @staticmethod
    def clip_grad_value(
        parameters: List[torch.nn.Parameter],
        clip_value: float = 1.0,
    ) -> None:
        """
        Clip gradients by value.
        
        Args:
            parameters: Model parameters
            clip_value: Maximum gradient value
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)
    
    @staticmethod
    def adaptive_gradient_clipping(
        parameters: List[torch.nn.Parameter],
        clip_factor: float = 0.01,
        eps: float = 1e-3,
    ) -> None:
        """
        Implement adaptive gradient clipping.
        
        Args:
            parameters: Model parameters
            clip_factor: Clipping factor
            eps: Small constant for numerical stability
            
        Reference:
            Gradient Clipping Improves Training of Transformers (Chen et al., 2020)
        """
        # Calculate per-parameter threshold
        for param in parameters:
            if param.grad is not None:
                # Calculate the L2 norm of parameters
                param_norm = param.data.norm(2)
                grad_norm = param.grad.data.norm(2)
                
                # Skip if gradient is zero
                if grad_norm == 0 or param_norm == 0:
                    continue
                
                # Calculate clipping threshold
                max_norm = clip_factor * param_norm
                
                # Clip gradients
                if grad_norm > max_norm:
                    scaling_factor = max_norm / (grad_norm + eps)
                    param.grad.data.mul_(scaling_factor)


def configure_weight_decay_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Optimizer:
    """
    Configure optimizer with weight decay applied selectively.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        beta1: Beta1 for AdamW
        beta2: Beta2 for AdamW
        eps: Epsilon for AdamW
        
    Returns:
        Optimizer: Configured optimizer
        
    Reference:
        Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
    """
    # Prepare parameter groups
    decay_params = []
    no_decay_params = []
    
    # Patterns for parameters that should not have weight decay applied
    no_decay_patterns = [
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "norm",
    ]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(pattern in name for pattern in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create optimizer with two parameter groups
    parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )


def create_cosine_annealing_with_warmup_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Custom cosine annealing scheduler with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate ratio at the end
        last_epoch: Last epoch
        
    Returns:
        LambdaLR: Learning rate scheduler
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decay = min_lr_ratio + (1.0 - min_lr_ratio) * cos_decay
        
        return decay
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
