"""
Legal document analyzer using the Artemis framework.
"""

import logging
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from artemis.utils.hybrid_adapter import add_hybrid_adapter_to_model, HybridAdapterConfig
from artemis.utils.pruning import apply_structured_pruning
from artemis.utils.efficiency import optimize_transformer_efficiency

logger = logging.getLogger(__name__)

class LegalDocumentAnalyzer:
    """
    Legal document analyzer for extracting information and insights from legal documents.
    """
    
    def __init__(self, config: Union[Dict[str, Any], str, Path]):
        """
        Initialize the Legal Document Analyzer.
        
        Args:
            config: Configuration dictionary or path to config file
        """
        # Load configuration
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._initialize_model()
        self._initialize_extractors()
        self._initialize_analyzers()
    
    def _initialize_model(self):
        """Initialize the language model with Artemis optimizations."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Initializing model from {self.config['model']['model_path']}")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['model_path']
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_path']
        )
        
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
            logger.info("Applied hybrid LoRA-Adapter")
        
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
        
        # Apply quantization if enabled
        if self.config['optimization']['quantization'] == 'int8':
            try:
                import bitsandbytes as bnb
                self.model = bnb.nn.modules.Params8bit.quantize_model_8bit(self.model)
                logger.info("Applied 8-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not installed, skipping quantization")
        
        logger.info("Model initialization complete")
    
    def _initialize_extractors(self):
        """Initialize information extractors."""
        self.extractors = {}
        
        # Initialize clause extractor
        from .extractors.clause_extractor import ClauseExtractor
        self.extractors['clause'] = ClauseExtractor(
            clause_types=self.config['extraction']['clause_types'],
            method=self.config['extraction']['extraction_method'],
            confidence_threshold=self.config['extraction']['confidence_threshold']
        )
        
        # Initialize entity extractor
        from .extractors.entity_extractor import EntityExtractor
        self.extractors['entity'] = EntityExtractor(
            entity_types=self.config['extraction']['entity_types'],
            method=self.config['extraction']['extraction_method'],
            confidence_threshold=self.config['extraction']['confidence_threshold']
        )
        
        # Initialize document structure analyzer
        from .extractors.structure_analyzer import DocumentStructureAnalyzer
        self.extractors['structure'] = DocumentStructureAnalyzer(
            document_types=self.config['legal_extensions']['document_types']
        )
        
        logger.info("Extractors initialized")
    
    def _initialize_analyzers(self):
        """Initialize analysis components."""
        self.analyzers = {}
        
        # Initialize risk analyzer
        from .analyzers.risk_analyzer import RiskAnalyzer
        self.analyzers['risk'] = RiskAnalyzer(
            risk_levels=self.config['analysis']['risk_assessment_levels'],
            jurisdiction=self.config['legal_extensions']['jurisdiction']
        )
        
        # Initialize cross-reference analyzer
        if self.config['analysis']['cross_reference_enabled']:
            from .analyzers.cross_reference_analyzer import CrossReferenceAnalyzer
            self.analyzers['cross_reference'] = CrossReferenceAnalyzer(
                similarity_threshold=self.config['analysis']['semantic_similarity_threshold']
            )
        
        # Initialize summarizer
        from .analyzers.summarizer import LegalDocumentSummarizer
        self.analyzers['summarizer'] = LegalDocumentSummarizer(
            max_length=self.config['summarization']['max_summary_length'],
            include_key_clauses=self.config['summarization']['include_key_clauses'],
            include_parties=self.config['summarization']['include_parties'],
            include_dates=self.config['summarization']['include_dates'],
            include_monetary_values=self.config['summarization']['include_monetary_values'],
            include_issues=self.config['summarization']['include_issues']
        )
        
        # Initialize citation linker
        if self.config['legal_extensions']['enable_citation_linking']:
            from .analyzers.citation_linker import CitationLinker
            self.analyzers['citation'] = CitationLinker(
                jurisdiction=self.config['legal_extensions']['jurisdiction']
            )
        
        logger.info("Analyzers initialized")
    
    def _preprocess_document(self, document_text: str) -> str:
        """
        Preprocess document text.
        
        Args:
            document_text: Raw document text
        
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(document_text.split())
        
        # Normalize section headers
        # Additional preprocessing logic as needed
        
        return text
    
    def _generate_from_model(self, prompt: str, max_tokens: int = None) -> str:
        """
        Generate text from the language model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        if max_tokens is None:
            max_tokens = self.config['inference']['max_tokens']
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=max_tokens + inputs.input_ids.shape[1],
                temperature=self.config['inference']['temperature'],
                top_p=self.config['inference']['top_p'],
                top_k=self.config['inference']['top_k'],
                repetition_penalty=self.config['inference']['repetition_penalty'],
                do_sample=True,
                num_return_sequences=1,
            )
        
        # Decode
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def extract_information(self, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from a legal document.
        
        Args:
            document_text: Text content of the document
            metadata: Document metadata
        
        Returns:
            Extracted information
        """
        # Preprocess document
        processed_text = self._preprocess_document(document_text)
        
        # Extract document structure
        structure = self.extractors['structure'].extract(processed_text)
        
        # Extract entities
        entities = self.extractors['entity'].extract(processed_text)
        
        # Extract clauses
        clauses = self.extractors['clause'].extract(processed_text, structure)
        
        # Organize results
        results = {
            "document_type": structure.get("document_type", "unknown"),
            "document_structure": structure,
            "entities": entities,
            "clauses": clauses,
            "metadata": metadata
        }
        
        return results
    
    def analyze_risks(self, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze risks in a legal document.
        
        Args:
            document_text: Text content of the document
            metadata: Document metadata
        
        Returns:
            Risk analysis results
        """
        # First extract information
        extraction_results = self.extract_information(document_text, metadata)
        
        # Analyze risks
        risks = self.analyzers['risk'].analyze(
            document_text, 
            extraction_results['clauses'],
            extraction_results['document_type']
        )
        
        # Organize results
        results = {
            "document_type": extraction_results["document_type"],
            "risks": risks,
            "clauses_analyzed": len(extraction_results['clauses']),
            "overall_risk_level": self._calculate_overall_risk(risks)
        }
        
        return results
    
    def _calculate_overall_risk(self, risks: List[Dict[str, Any]]) -> str:
        """
        Calculate overall risk level based on individual risks.
        
        Args:
            risks: List of identified risks
            
        Returns:
            Overall risk level
        """
        if not risks:
            return "low"
        
        # Count risks by severity
        risk_counts = {"high": 0, "medium": 0, "low": 0, "informational": 0}
        for risk in risks:
            severity = risk.get("severity", "informational").lower()
            if severity in risk_counts:
                risk_counts[severity] += 1
        
        # Determine overall risk
        if risk_counts["high"] > 0:
            return "high"
        elif risk_counts["medium"] > 0:
            return "medium"
        elif risk_counts["low"] > 0:
            return "low"
        else:
            return "informational"
    
    def generate_summary(self, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of a legal document.
        
        Args:
            document_text: Text content of the document
            metadata: Document metadata
        
        Returns:
            Document summary
        """
        # First extract information
        extraction_results = self.extract_information(document_text, metadata)
        
        # Generate summary
        summary = self.analyzers['summarizer'].summarize(
            document_text,
            extraction_results
        )
        
        # Organize results
        results = {
            "document_type": extraction_results["document_type"],
            "summary": summary,
            "key_entities": self._extract_key_entities(extraction_results['entities']),
            "word_count": len(document_text.split()),
            "summary_word_count": len(summary.split())
        }
        
        return results
    
    def _extract_key_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """
        Extract key entities for summary.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Key entities
        """
        key_entities = {}
        
        # Extract parties (organizations and persons)
        if 'organization' in entities:
            key_entities['organizations'] = [e['text'] for e in entities['organization'][:5]]
        
        if 'person' in entities:
            key_entities['persons'] = [e['text'] for e in entities['person'][:5]]
        
        # Extract dates
        if 'date' in entities:
            key_entities['dates'] = [e['text'] for e in entities['date'][:5]]
        
        # Extract monetary values
        if 'monetary_value' in entities:
            key_entities['monetary_values'] = [e['text'] for e in entities['monetary_value'][:5]]
        
        return key_entities
    
    def analyze(self, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform full analysis of a legal document.
        
        Args:
            document_text: Text content of the document
            metadata: Document metadata
        
        Returns:
            Complete analysis results
        """
        # Extract information
        extraction_results = self.extract_information(document_text, metadata)
        
        # Analyze risks
        risks = self.analyzers['risk'].analyze(
            document_text, 
            extraction_results['clauses'],
            extraction_results['document_type']
        )
        
        # Generate summary
        summary = self.analyzers['summarizer'].summarize(
            document_text,
            extraction_results
        )
        
        # Add citation links if enabled
        citations = []
        if 'citation' in self.analyzers:
            citations = self.analyzers['citation'].find_citations(document_text)
        
        # Analyze cross-references if enabled
        cross_references = []
        if 'cross_reference' in self.analyzers:
            cross_references = self.analyzers['cross_reference'].analyze(
                extraction_results['clauses'],
                document_text
            )
        
        # Organize complete results
        results = {
            "document_type": extraction_results["document_type"],
            "subtype": extraction_results["document_structure"].get("subtype", ""),
            "parties": self._extract_parties(extraction_results['entities']),
            "key_dates": self._extract_dates(extraction_results['entities']),
            "key_terms": self._extract_key_terms(extraction_results['clauses']),
            "clauses": self._format_clauses(extraction_results['clauses']),
            "potential_issues": risks,
            "overall_risk_level": self._calculate_overall_risk(risks),
            "citations": citations,
            "cross_references": cross_references,
            "summary": summary,
            "metadata": metadata
        }
        
        return results
    
    def _extract_parties(self, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Extract and format party information."""
        parties = []
        
        # Extract organizations
        if 'organization' in entities:
            for org in entities['organization']:
                if org.get('role'):
                    parties.append({
                        "name": org['text'],
                        "type": "organization",
                        "role": org['role']
                    })
        
        # Extract persons
        if 'person' in entities:
            for person in entities['person']:
                if person.get('role'):
                    parties.append({
                        "name": person['text'],
                        "type": "person",
                        "role": person['role']
                    })
        
        return parties
    
    def _extract_dates(self, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Extract and format date information."""
        key_dates = []
        
        if 'date' in entities:
            for date in entities['date']:
                if date.get('date_type'):
                    key_dates.append({
                        "type": date['date_type'],
                        "date": date['normalized'] if 'normalized' in date else date['text']
                    })
        
        return key_dates
    
    def _extract_key_terms(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key terms from clauses."""
        key_terms = []
        
        # Important clause types to include as key terms
        key_clause_types = [
            "payment", "term", "termination", "renewal", "governing_law",
            "dispute_resolution", "limitation_of_liability"
        ]
        
        for clause in clauses:
            if clause['type'] in key_clause_types:
                key_terms.append({
                    "type": clause['type'],
                    "description": clause.get('summary', clause['content'][:100] + "..."),
                    "location": clause.get('location', {})
                })
        
        return key_terms
    
    def _format_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format clauses for output."""
        formatted_clauses = []
        
        for clause in clauses:
            formatted_clause = {
                "type": clause['type'],
                "content": clause['content'][:500] + "..." if len(clause['content']) > 500 else clause['content'],
                "location": clause.get('location', {})
            }
            
            # Add risk assessment if available
            if 'risk_assessment' in clause:
                formatted_clause['risk_assessment'] = clause['risk_assessment']
            
            formatted_clauses.append(formatted_clause)
        
        return formatted_clauses
    
    def compare_documents(self, document1: str, document2: str) -> Dict[str, Any]:
        """
        Compare two legal documents.
        
        Args:
            document1: First document text
            document2: Second document text
        
        Returns:
            Comparison results
        """
        # Extract information from both documents
        results1 = self.extract_information(document1, {})
        results2 = self.extract_information(document2, {})
        
        # Identify similarities and differences
        comparison = {
            "document_types": {
                "document1": results1["document_type"],
                "document2": results2["document_type"]
            },
            "clause_comparison": self._compare_clauses(results1["clauses"], results2["clauses"]),
            "entity_comparison": self._compare_entities(results1["entities"], results2["entities"]),
            "summary": self._generate_comparison_summary(results1, results2)
        }
        
        return comparison
    
    def _compare_clauses(self, clauses1: List[Dict[str, Any]], clauses2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare clauses between documents."""
        # Implementation would use semantic similarity to match clauses
        # and identify differences
        pass
    
    def _compare_entities(self, entities1: Dict[str, List[Dict[str, Any]]], entities2: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compare entities between documents."""
        # Implementation would identify common and unique entities
        pass
    
    def _generate_comparison_summary(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> str:
        """Generate a summary of document comparison."""
        # Implementation would create a natural language summary
        # of key similarities and differences
        pass
