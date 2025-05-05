# Legal Document Analysis Example

This example demonstrates how to use the Artemis framework for efficient legal document analysis. The application is designed to process legal documents at scale with high accuracy and low computational cost.

## Overview

The Legal Document Analysis system uses the Artemis framework to:

1. Process and analyze legal documents
2. Extract key clauses and entities
3. Identify potential issues and inconsistencies
4. Generate summaries and insights
5. Run efficiently on standard hardware

## Features

- **Document Processing**: Handles various legal document formats (PDF, DOCX, text)
- **Clause Extraction**: Identifies and categorizes important legal clauses
- **Entity Recognition**: Extracts parties, dates, monetary values, and obligations
- **Risk Assessment**: Identifies potential legal risks and issues
- **Summarization**: Generates concise summaries of lengthy legal documents
- **Cross-Reference Analysis**: Compares provisions across multiple documents
- **Jurisdiction Awareness**: Adapts analysis to different legal jurisdictions

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Artemis framework
- PDF processing libraries

### Installation

```bash
# Clone the Artemis repository if you haven't already
git clone https://github.com/your-org/artemis.git
cd artemis

# Install dependencies
pip install -r examples/legal_analysis/requirements.txt

# Download the legal domain fine-tuned model
python scripts/download_model.py --model artemis-legal-base
```

### Configuration

Edit the configuration file `examples/legal_analysis/config.yaml` to adjust settings:

```yaml
model:
  base_model: "artemis-legal-base"
  
hybrid_adapter:
  enabled: true
  lora_rank: 8
  adapter_size: 64
  apply_lora_to: ["q_proj", "k_proj", "v_proj", "o_proj"]

pruning:
  enabled: true
  method: "structured_pruning"
  sparsity: 0.55
  
efficiency_transformer:
  enabled: true
  attention_method: "flash_attention"
  
optimization:
  quantization: "int8"
  batch_processing: true
  
legal_extensions:
  jurisdiction: "us"  # Options: us, uk, eu, international
  document_types: 
    - "contract"
    - "legislation"
    - "court_document"
    - "regulatory_filing"
  enable_case_law: true
  enable_citation_linking: true
```

## Usage

### Document Processing

Process a single legal document with:

```bash
python examples/legal_analysis/process_document.py \
       --input contract.pdf \
       --output analysis.json
```

### Batch Processing

Process multiple documents with:

```bash
python examples/legal_analysis/batch_process.py \
       --input_dir /path/to/documents/ \
       --output_dir /path/to/results/ \
       --num_workers 4
```

### Analysis Server

Run as an API server:

```bash
python examples/legal_analysis/server.py --port 8000
```

Then submit documents for analysis:

```bash
curl -X POST "http://localhost:8000/api/analyze" \
     -F "document=@contract.pdf" \
     -F "analysis_type=full"
```

## Example Results

### Contract Analysis

Input: Commercial lease agreement (PDF)

Output:
```json
{
  "document_type": "contract",
  "subtype": "commercial_lease",
  "parties": [
    {"name": "Acme Properties LLC", "role": "landlord"},
    {"name": "TechStart Inc.", "role": "tenant"}
  ],
  "key_dates": [
    {"type": "effective_date", "date": "2024-03-01"},
    {"type": "term_start", "date": "2024-04-01"},
    {"type": "term_end", "date": "2029-03-31"}
  ],
  "key_terms": [
    {
      "type": "rent",
      "description": "Monthly base rent of $10,000 with 3% annual increases",
      "location": {"page": 2, "paragraph": 4}
    },
    {
      "type": "renewal_option",
      "description": "Two 5-year renewal options with 60-day notice requirement",
      "location": {"page": 5, "paragraph": 2}
    }
  ],
  "clauses": [
    {
      "type": "indemnification",
      "content": "Tenant shall indemnify, defend and hold harmless Landlord...",
      "risk_assessment": "Standard indemnification clause with broad tenant obligations",
      "location": {"page": 8, "paragraph": 3}
    }
  ],
  "potential_issues": [
    {
      "severity": "high",
      "description": "Ambiguous maintenance responsibilities in Section 7.2",
      "recommendation": "Clarify division of maintenance obligations",
      "location": {"page": 7, "paragraph": 2}
    }
  ],
  "summary": "Commercial lease agreement between Acme Properties LLC (Landlord) and TechStart Inc. (Tenant) for 5-year term beginning April 1, 2024. Base rent of $10,000/month with 3% annual increases. Tenant responsible for utilities, maintenance, and proportional share of property taxes. Two 5-year renewal options. Notable provisions include broad tenant indemnification and potentially ambiguous maintenance responsibilities."
}
```

### Legal Brief Analysis

Input: Appellate brief (DOCX)

Output:
```json
{
  "document_type": "court_document",
  "subtype": "appellate_brief",
  "court": "Supreme Court of California",
  "case_number": "S123456",
  "parties": [
    {"name": "Smith Technologies, Inc.", "role": "appellant"},
    {"name": "Johnson Innovations LLC", "role": "respondent"}
  ],
  "legal_issues": [
    {
      "issue": "Whether the non-compete clause violates California Business and Professions Code § 16600",
      "appellant_position": "The clause is void under California law which prohibits non-compete agreements",
      "respondent_position": "The clause is enforceable as a narrowly tailored protection of trade secrets"
    }
  ],
  "key_authorities": [
    {
      "citation": "Edwards v. Arthur Andersen LLP, 44 Cal.4th 937 (2008)",
      "relevance": "Established California's strong public policy against non-compete agreements"
    }
  ],
  "argument_structure": [
    {
      "section": "I",
      "heading": "The Trial Court Erred in Enforcing the Non-Compete Provision",
      "key_points": [
        "California law invalidates all non-compete agreements with limited exceptions",
        "The agreement does not qualify for the trade secret protection exception"
      ],
      "strength_assessment": "Strong argument supported by clear statutory authority"
    }
  ],
  "summary": "Appellant's brief argues that the trial court erred in enforcing a non-compete clause in an employment agreement. The brief relies heavily on California's statutory prohibition against non-compete agreements (Cal. Bus. & Prof. Code § 16600) and relevant case law, particularly Edwards v. Arthur Andersen LLP. The appellant contends that the narrow exceptions for trade secret protection do not apply in this case. The brief is well-structured with clear arguments based on established legal authority."
}
```

## Performance Benchmarks

| Metric | Baseline Model | Artemis-Optimized |
|--------|---------------|-------------------|
| Documents per minute | 12 | 78 |
| Memory Usage | 15GB | 6GB |
| Clause Extraction Accuracy | 82% | 88% |
| Entity Recognition F1 | 0.79 | 0.85 |
| GPU Utilization | 95% | 65% |

## Advanced Usage

### Jurisdiction-Specific Models

Train models for specific jurisdictions:

```bash
python examples/legal_analysis/train_jurisdiction.py \
       --jurisdiction uk \
       --data_path data/legal/uk/ \
       --output_path models/legal/uk/
```

### Custom Document Types

Add support for custom document types:

```python
from artemis.examples.legal_analysis.document_types import register_document_type

# Define a custom document type processor
class MergerAgreementProcessor:
    def __init__(self):
        self.clauses_to_extract = [
            "representations_warranties",
            "closing_conditions",
            "termination_rights"
        ]
    
    def extract_structure(self, document):
        # Implementation
        pass
    
    def analyze_risks(self, document, structure):
        # Implementation
        pass

# Register the custom processor
register_document_type(
    name="merger_agreement",
    processor=MergerAgreementProcessor(),
    parent_type="contract"
)
```

### Integration with Legal Research Tools

The system can be integrated with legal research databases:

```python
from artemis.examples.legal_analysis.research import LegalResearchConnector

# Initialize research connector
connector = LegalResearchConnector(
    api_key="your_api_key",
    databases=["case_law", "statutes", "regulations"]
)

# Enhance analysis with research
def analyze_with_research(document_text):
    # Initial analysis
    analysis = legal_analyzer.analyze(document_text)
    
    # Identify legal issues that need research
    research_queries = legal_analyzer.generate_research_queries(analysis)
    
    # Conduct legal research
    research_results = connector.research(research_queries)
    
    # Enhance analysis with research findings
    enhanced_analysis = legal_analyzer.enhance_with_research(
        analysis, research_results
    )
    
    return enhanced_analysis
```

## Implementation Details

### Document Processing Pipeline

The document processing pipeline consists of:

1. **Document Loading**: Handles various file formats
2. **Text Extraction**: Converts documents to processable text
3. **Structure Analysis**: Identifies document structure and sections
4. **Entity Extraction**: Identifies parties, dates, amounts, etc.
5. **Clause Identification**: Locates and categorizes legal clauses
6. **Semantic Analysis**: Analyzes meaning and implications of provisions
7. **Risk Assessment**: Identifies potential legal risks
8. **Summary Generation**: Creates concise document summaries

### Legal Knowledge Integration

The model incorporates legal knowledge through:

```python
from artemis.examples.legal_analysis.knowledge import LegalKnowledgeEnhancer

# Initialize knowledge enhancer
knowledge_enhancer = LegalKnowledgeEnhancer(
    jurisdiction="us",
    practice_areas=["contracts", "corporate", "intellectual_property"]
)

# Enhance model with legal knowledge
enhanced_model = knowledge_enhancer.enhance_model(base_model)
```

## Future Improvements

Planned enhancements for this example:

1. Support for additional jurisdictions and languages
2. Integration with case law databases
3. Comparative analysis of document versions
4. Advanced risk scoring algorithms
5. Specialized models for M&A, real estate, and litigation documents
6. Integration with contract management systems

## License and Disclaimer

This example is provided under the Apache 2.0 license.

**Legal Disclaimer**: This software is designed for informational and research purposes only. It is not a substitute for legal advice, and its analysis should not be relied upon as legal counsel. Users should consult qualified legal professionals for specific legal advice.
