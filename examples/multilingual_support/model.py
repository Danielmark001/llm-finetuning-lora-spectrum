"""
Multilingual Customer Support model implementation using the Artemis framework.
This module provides the core model architecture and inference functionality.
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

from artemis.utils.hybrid_adapter import add_hybrid_adapter_to_model, HybridAdapterConfig
from artemis.utils.efficiency import optimize_transformer_efficiency
from artemis.utils.evaluation import evaluate_model_on_dataset
from artemis.multilingual.language_detection import LanguageDetector
from artemis.multilingual.adapters import MultilingualAdapterConfig, add_multilingual_adapters
from artemis.multilingual.terminology import TerminologyManager

logger = logging.getLogger(__name__)

class MultilingualSupportModel:
    """
    Multilingual customer support model using Artemis efficiency techniques.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the Multilingual Support model with the specified configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize base model
        logger.info(f"Initializing Multilingual Support model from {self.config['model']['base_model']}")
        self.model = self._initialize_base_model()
        
        # Create language map
        self.languages = {lang['code']: lang for lang in self.config['languages']}
        self.default_language = next((lang['code'] for lang in self.config['languages'] if lang.get('is_default')), 'en')
        
        # Apply Artemis optimizations
        self._apply_optimizations()
        
        # Load support extensions
        self._load_support_extensions()
        
        # Set initial language to default
        self.current_language = self.default_language
        if hasattr(self, 'language_detector'):
            logger.info(f"Language detection enabled, initial language set to {self.current_language}")

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
        # Apply multilingual adapters if enabled
        if self.config['hybrid_adapter']['enabled']:
            # Configure language-specific adapters
            adapter_config = MultilingualAdapterConfig(
                base_adapter_size=self.config['hybrid_adapter']['adapter_size'],
                lora_rank=self.config['hybrid_adapter']['lora_rank'],
                language_specific_ranks=self.config.get('language_specific_ranks', {}),
                shared_layers=self.config.get('shared_layers', []),
                language_specific_layers=self.config.get('language_specific_layers', []),
                target_modules=self.config['hybrid_adapter']['target_modules'],
                apply_lora_to=self.config['hybrid_adapter']['apply_lora_to'],
                shared_parameters=self.config['hybrid_adapter']['shared_parameters'],
                scaling_factor=self.config['hybrid_adapter']['scaling_factor']
            )
            
            # Get supported language codes
            supported_languages = [lang['code'] for lang in self.config['languages']]
            
            # Add adapters to model
            self.model = add_multilingual_adapters(
                self.model, 
                adapter_config, 
                languages=supported_languages
            )
            logger.info(f"Applied multilingual adapters for {len(supported_languages)} languages")
        
        # Apply efficiency transformer optimizations if enabled
        if self.config['efficiency_transformer']['enabled']:
            optimize_transformer_efficiency(
                self.model,
                attention_method=self.config['efficiency_transformer']['attention_method'],
                kv_cache_method=self.config['efficiency_transformer']['kv_cache_method'],
                memory_efficient=self.config['efficiency_transformer']['memory_efficient']
            )
            logger.info("Applied efficiency transformer optimizations")
            
        # Apply quantization if enabled
        if self.config['optimization'].get('quantization'):
            from artemis.utils.quantization import quantize_model
            quantize_model(
                self.model,
                method=self.config['optimization']['quantization']
            )
            logger.info(f"Applied {self.config['optimization']['quantization']} quantization")
            
        # Apply pruning if enabled
        if self.config['optimization'].get('pruning', {}).get('enabled', False):
            from artemis.utils.pruning import apply_structured_pruning
            apply_structured_pruning(
                self.model,
                sparsity=self.config['optimization']['pruning']['sparsity'],
                method=self.config['optimization']['pruning']['method']
            )
            logger.info(f"Applied {self.config['optimization']['pruning']['method']} with {self.config['optimization']['pruning']['sparsity']} sparsity")
    
    def _load_support_extensions(self):
        """Load customer support domain-specific extensions."""
        # Language detection
        if self.config['support_extensions']['enable_language_detection']:
            from artemis.multilingual.language_detection import LanguageDetector
            supported_languages = [lang['code'] for lang in self.config['languages']]
            self.language_detector = LanguageDetector(supported_languages=supported_languages)
            logger.info("Loaded language detector")
        
        # Terminology database
        if self.config['support_extensions']['enable_terminology_database']:
            from artemis.multilingual.terminology import TerminologyManager
            supported_languages = [lang['code'] for lang in self.config['languages']]
            self.terminology_manager = TerminologyManager(
                terminology_database=self.config['support_extensions']['terminology_database_path'],
                enabled_languages=supported_languages
            )
            logger.info("Loaded terminology manager")
        
        # Sentiment analysis
        if self.config['support_extensions']['enable_sentiment_analysis']:
            from artemis.multilingual.sentiment import SentimentAnalyzer
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("Loaded sentiment analyzer")
        
        # Intent detection
        if self.config['support_extensions']['enable_intent_detection']:
            from artemis.multilingual.intent import IntentDetector
            self.intent_detector = IntentDetector(domain=self.config['support_extensions']['domain'])
            logger.info(f"Loaded intent detector for {self.config['support_extensions']['domain']} domain")
        
        # Response templates
        if self.config['support_extensions']['response_templates_enabled']:
            from artemis.multilingual.templates import ResponseTemplateManager
            supported_languages = [lang['code'] for lang in self.config['languages']]
            self.template_manager = ResponseTemplateManager(
                templates_path=self.config['support_extensions']['response_templates_path'],
                enabled_languages=supported_languages
            )
            logger.info("Loaded response template manager")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        if hasattr(self, 'language_detector'):
            detected_lang = self.language_detector.detect(text)
            if detected_lang in self.languages:
                return detected_lang
        
        # Fall back to default language if detection fails or is not available
        return self.default_language
    
    def activate_language_adapter(self, language_code: str):
        """
        Activate the adapter for the specified language.
        
        Args:
            language_code: The language code to activate
        """
        if language_code not in self.languages:
            logger.warning(f"Language {language_code} not supported, using {self.default_language}")
            language_code = self.default_language
        
        # Skip if already activated
        if language_code == self.current_language:
            return
        
        # Activate language adapter
        self.model.activate_language_adapter(language_code)
        self.current_language = language_code
        logger.info(f"Activated adapter for {language_code}")
    
    def preprocess_query(self, query: str, language: Optional[str] = None) -> Tuple[str, str]:
        """
        Preprocess a query before passing to the model.
        
        Args:
            query: The raw query
            language: Optional language code (if None, will be detected)
            
        Returns:
            Tuple of (processed query, language code)
        """
        # Detect language if not provided
        if language is None:
            language = self.detect_language(query)
        
        # Activate language adapter
        self.activate_language_adapter(language)
        
        # Apply intent detection if enabled
        intent = None
        entities = None
        if hasattr(self, 'intent_detector'):
            intent, entities = self.intent_detector.detect(query, language)
            
        # Check for template match if enabled
        if hasattr(self, 'template_manager') and intent is not None:
            template = self.template_manager.find_template(intent, entities, language)
            if template:
                # Store for response generation
                self._current_template = template
                self._current_entities = entities
        
        return query, language
    
    def postprocess_response(self, response: str, language: str) -> str:
        """
        Postprocess the model's response.
        
        Args:
            response: The raw model response
            language: The language code
            
        Returns:
            Enhanced response
        """
        # Ensure terminology consistency if enabled
        if hasattr(self, 'terminology_manager'):
            response = self.terminology_manager.apply_terminology(response, language)
        
        return response
    
    def generate(self, query: str, language: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response to a query.
        
        Args:
            query: The customer query
            language: Optional language code (if None, will be detected)
            session_id: Optional session ID for conversation context
            
        Returns:
            Dictionary with response and metadata
        """
        # Clear template state
        self._current_template = None
        self._current_entities = None
        
        # Preprocess the query
        processed_query, language_code = self.preprocess_query(query, language)
        
        # Check for template match first
        if self._current_template is not None:
            response = self.template_manager.fill_template(
                self._current_template, 
                self._current_entities
            )
            logger.info(f"Using template response for {language_code}")
        else:
            # No template match, generate response with model
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
        enhanced_response = self.postprocess_response(response, language_code)
        
        # Analyze sentiment if enabled
        sentiment = None
        if hasattr(self, 'sentiment_analyzer'):
            sentiment = self.sentiment_analyzer.analyze(query, language_code)
        
        # Update session info if provided
        if session_id:
            self._update_session(session_id, language_code)
        
        return {
            "response": enhanced_response,
            "language": language_code,
            "sentiment": sentiment,
            "used_template": self._current_template is not None
        }
    
    def batch_generate(self, queries: List[str], languages: Optional[List[str]] = None, session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of queries.
        
        Args:
            queries: List of customer queries
            languages: Optional list of language codes (if None, will be detected for each)
            session_ids: Optional list of session IDs for conversation context
            
        Returns:
            List of dictionaries with responses and metadata
        """
        if languages is None:
            languages = [None] * len(queries)
            
        if session_ids is None:
            session_ids = [None] * len(queries)
            
        results = []
        
        # Process in mini-batches based on language
        language_batches = {}
        for i, (query, lang, session_id) in enumerate(zip(queries, languages, session_ids)):
            # Detect language if not provided
            if lang is None:
                lang = self.detect_language(query)
                
            # Group by language for batch processing
            if lang not in language_batches:
                language_batches[lang] = []
            language_batches[lang].append((i, query, session_id))
        
        # Process each language batch
        for lang, batch in language_batches.items():
            indices, batch_queries, batch_session_ids = zip(*batch)
            
            # Activate language adapter once per batch
            self.activate_language_adapter(lang)
            
            # Process each query in the batch
            batch_results = []
            for query, session_id in zip(batch_queries, batch_session_ids):
                result = self.generate(query, lang, session_id)
                batch_results.append(result)
            
            # Assign results back to original order
            for i, result in zip(indices, batch_results):
                while len(results) <= i:
                    results.append(None)
                results[i] = result
        
        return results
    
    def _update_session(self, session_id: str, language: str):
        """
        Update session information.
        
        Args:
            session_id: Session identifier
            language: Language code used in this interaction
        """
        # In a real implementation, this would store session state
        # For now, we just log that we would update the session
        logger.debug(f"Updated session {session_id} with language {language}")
    
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
    
    def evaluate(self, test_data_path: Optional[Union[str, Path]] = None, languages: Optional[List[str]] = None):
        """
        Evaluate the model on multilingual test data.
        
        Args:
            test_data_path: Path to test data (defaults to config path)
            languages: List of languages to evaluate (defaults to all supported)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if test_data_path is None:
            test_data_path = self.config['data']['test_data']
            
        if languages is None:
            languages = [lang['code'] for lang in self.config['languages']]
        
        results = {}
        
        # Evaluate on each language
        for lang in languages:
            lang_results = evaluate_model_on_dataset(
                self.model,
                test_data_path,
                metrics=self.config['evaluation']['metrics'],
                language=lang,
                device=self.device
            )
            results[lang] = lang_results
            
        # Calculate cross-lingual consistency if needed
        if 'cross_lingual_consistency' in self.config['evaluation']['metrics']:
            from artemis.multilingual.evaluation import measure_cross_lingual_consistency
            consistency = measure_cross_lingual_consistency(
                self.model,
                test_data_path,
                languages=languages,
                device=self.device
            )
            results['cross_lingual_consistency'] = consistency
        
        return results


def load_multilingual_support_model(model_path: Union[str, Path]) -> MultilingualSupportModel:
    """
    Load a saved Multilingual Support model.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Loaded MultilingualSupportModel instance
    """
    model_path = Path(model_path)
    config_path = model_path / "config.yaml"
    
    model = MultilingualSupportModel(config_path)
    logger.info(f"Loaded Multilingual Support model from {model_path}")
    
    return model
