"""
Medical QA model implementation using the Artemis framework.
This module provides the core model architecture and inference functionality.
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from artemis.utils.hybrid_adapter import add_hybrid_adapter_to_model, HybridAdapterConfig
from artemis.utils.pruning import apply_structured_pruning
from artemis.utils.efficiency import optimize_transformer_efficiency
from artemis.utils.evaluation import evaluate_model_on_dataset

logger = logging.getLogger(__name__)

class MedicalQAModel:
    """
    Medical domain question-answering model using Artemis efficiency techniques.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the Medical QA model with the specified configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize base model
        logger.info(f"Initializing Medical QA model from {self.config['model']['base_model']}")
        self.model = self._initialize_base_model()
        
        # Apply Artemis optimizations
        self._apply_optimizations()
        
        # Load medical extensions
        self._load_medical_extensions()

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_base_model(self):
        """Initialize the base model from configuration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = self.config['model']['model_path']
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move model to appropriate device
        model = model.to(self.device)
        return model
    
    def _apply_optimizations(self):
        """Apply Artemis optimizations to the base model."""
        # Apply hybrid adapter if enabled
        if self.config['hybrid_adapter']['enabled']:
            adapter_config = HybridAdapterConfig(
                lora_rank=self.config['hybrid_adapter']['lora_rank'],
                adapter_size=self.config['hybrid_adapter']['adapter_size'],
                target_modules=self.config['hybrid_adapter']['target_modules'],
                apply_lora_to=self.config['hybrid_adapter']['apply_lora_to'],
                shared_parameters=self.config['hybrid_adapter']['shared_parameters'],
                scaling_factor=self.config['hybrid_adapter']['scaling_factor']
            )
            self.model = add_hybrid_adapter_to_model(self.model, adapter_config)
            logger.info("Applied hybrid LoRA-Adapter to model")
        
        # Apply pruning if enabled
        if self.config['pruning']['enabled']:
            apply_structured_pruning(
                self.model,
                sparsity=self.config['pruning']['sparsity'],
                method=self.config['pruning']['method']
            )
            logger.info(f"Applied {self.config['pruning']['method']} with {self.config['pruning']['sparsity']} sparsity")
        
        # Apply efficiency transformer optimizations if enabled
        if self.config['efficiency_transformer']['enabled']:
            optimize_transformer_efficiency(
                self.model,
                attention_method=self.config['efficiency_transformer']['attention_method'],
                kv_cache_method=self.config['efficiency_transformer']['kv_cache_method'],
                memory_efficient=self.config['efficiency_transformer']['memory_efficient']
            )
            logger.info("Applied efficiency transformer optimizations")
    
    def _load_medical_extensions(self):
        """Load medical domain-specific extensions."""
        if self.config['medical_extensions']['terminology_recognition']:
            from .medical_terminology import MedicalTerminologyRecognizer
            self.term_recognizer = MedicalTerminologyRecognizer()
            logger.info("Loaded medical terminology recognizer")
        
        if self.config['medical_extensions']['evidence_linking']:
            from .evidence import EvidenceLinker
            self.evidence_linker = EvidenceLinker(
                sources=self.config['medical_extensions']['knowledge_sources']
            )
            logger.info("Loaded evidence linker")
        
        if self.config['medical_extensions']['entity_extraction']:
            from .entity_extraction import MedicalEntityExtractor
            self.entity_extractor = MedicalEntityExtractor()
            logger.info("Loaded medical entity extractor")
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a medical query before passing to the model.
        
        Args:
            query: The raw medical query
            
        Returns:
            Preprocessed query with recognized medical terminology
        """
        # Apply medical terminology recognition if enabled
        if hasattr(self, 'term_recognizer'):
            query = self.term_recognizer.process(query)
        
        # Extract and normalize medical entities if enabled
        if hasattr(self, 'entity_extractor'):
            entities = self.entity_extractor.extract(query)
            query = self.entity_extractor.enhance_query(query, entities)
        
        return query
    
    def postprocess_response(self, response: str, query: str) -> str:
        """
        Postprocess the model's response to enhance medical accuracy.
        
        Args:
            response: The raw model response
            query: The original query for context
            
        Returns:
            Enhanced response with evidence links and disclaimers
        """
        # Add evidence links if enabled
        if hasattr(self, 'evidence_linker'):
            response = self.evidence_linker.enhance_with_evidence(response, query)
        
        # Add medical disclaimer if configured
        if self.config['medical_extensions']['disclaimer_generation']:
            response = self._add_medical_disclaimer(response)
        
        return response
    
    def _add_medical_disclaimer(self, response: str) -> str:
        """Add an appropriate medical disclaimer to the response."""
        disclaimer = (
            "\n\nMedical Disclaimer: This information is for educational purposes only "
            "and not a substitute for professional medical advice. Consult a healthcare "
            "provider for diagnosis and treatment."
        )
        return response + disclaimer
    
    def generate(self, query: str) -> str:
        """
        Generate a response to a medical query.
        
        Args:
            query: The medical question
            
        Returns:
            Response with medical information
        """
        # Preprocess the query
        processed_query = self.preprocess_query(query)
        
        # Tokenize
        inputs = self.tokenizer(processed_query, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=self.config['inference']['max_tokens'],
                temperature=self.config['inference']['temperature'],
                top_p=self.config['inference']['top_p'],
                top_k=self.config['inference']['top_k'],
                repetition_penalty=self.config['inference']['repetition_penalty'],
                do_sample=True,
                num_return_sequences=1,
            )
        
        # Decode response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Strip the original query if it appears in the response
        if response.startswith(processed_query):
            response = response[len(processed_query):].strip()
        
        # Postprocess the response
        enhanced_response = self.postprocess_response(response, query)
        
        return enhanced_response
    
    def batch_generate(self, queries: List[str]) -> List[str]:
        """
        Generate responses for a batch of medical queries.
        
        Args:
            queries: List of medical questions
            
        Returns:
            List of responses with medical information
        """
        # Preprocess queries
        processed_queries = [self.preprocess_query(q) for q in queries]
        
        # Tokenize
        batch_inputs = self.tokenizer(
            processed_queries, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            batch_output_ids = self.model.generate(
                batch_inputs.input_ids,
                max_length=self.config['inference']['max_tokens'],
                temperature=self.config['inference']['temperature'],
                top_p=self.config['inference']['top_p'],
                top_k=self.config['inference']['top_k'],
                repetition_penalty=self.config['inference']['repetition_penalty'],
                do_sample=True,
                num_return_sequences=1,
            )
        
        # Decode responses
        responses = [
            self.tokenizer.decode(output_ids, skip_special_tokens=True)
            for output_ids in batch_output_ids
        ]
        
        # Postprocess responses
        enhanced_responses = [
            self.postprocess_response(response, query)
            for response, query in zip(responses, queries)
        ]
        
        return enhanced_responses
    
    def save(self, save_path: Union[str, Path]):
        """
        Save the model and configuration.
        
        Args:
            save_path: Directory to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        with open(save_path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def evaluate(self, test_data_path: Optional[Union[str, Path]] = None):
        """
        Evaluate the model on medical QA test data.
        
        Args:
            test_data_path: Path to test data (defaults to config path)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if test_data_path is None:
            test_data_path = self.config['data']['test_data']
        
        results = evaluate_model_on_dataset(
            self.model,
            test_data_path,
            metrics=self.config['evaluation']['metrics'],
            device=self.device
        )
        
        return results


def load_medical_qa_model(model_path: Union[str, Path]) -> MedicalQAModel:
    """
    Load a saved Medical QA model.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Loaded MedicalQAModel instance
    """
    model_path = Path(model_path)
    config_path = model_path / "config.yaml"
    
    model = MedicalQAModel(config_path)
    logger.info(f"Loaded Medical QA model from {model_path}")
    
    return model
