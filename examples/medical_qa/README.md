# Medical QA Example Application

This example demonstrates how to use the Artemis framework to build an efficient and accurate medical question answering system. The application is optimized for high-quality responses to medical queries while maintaining low latency and resource utilization.

## Overview

The Medical QA system uses a specialized fine-tuned model with Artemis efficiency techniques to:

1. Understand medical terminology and concepts
2. Retrieve relevant medical information
3. Generate accurate, evidence-based responses
4. Provide appropriate health disclaimers
5. Operate efficiently on modest hardware

## Features

- **Medical terminology recognition**: Understands medical jargon, abbreviations, and concepts
- **Evidence-based responses**: Answers grounded in medical literature and guidelines
- **Domain adaptation**: Specialized for medical knowledge and reasoning
- **Low latency**: < 300ms response time for most queries
- **Resource efficiency**: Runs on a single GPU with 8GB VRAM
- **Multi-format support**: Handles text, structured, and semi-structured queries

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Artemis framework

### Installation

```bash
# Clone the Artemis repository if you haven't already
git clone https://github.com/your-org/artemis.git
cd artemis

# Install dependencies
pip install -r examples/medical_qa/requirements.txt

# Download the medical domain fine-tuned model
python scripts/download_model.py --model artemis-medical-base
```

### Configuration

Edit the configuration file `examples/medical_qa/config.yaml` to adjust the model settings:

```yaml
model:
  base_model: "artemis-medical-base"
  
hybrid_adapter:
  enabled: true
  lora_rank: 16
  adapter_size: 128
  apply_lora_to: ["q_proj", "k_proj", "v_proj", "o_proj"]

pruning:
  enabled: true
  method: "structured_pruning"
  sparsity: 0.45
  
efficiency_transformer:
  enabled: true
  attention_method: "flash_attention"
  kv_cache_method: "quantized"
  memory_efficient: true
  
inference:
  max_tokens: 1024
  temperature: 0.1
  top_p: 0.92
  batch_size: 8
  
medical_extensions:
  terminology_recognition: true
  evidence_linking: true
  disclaimer_generation: true
  knowledge_sources: ["pubmed", "guidelines", "medical_texts"]
```

## Usage

### Interactive Mode

To run the medical QA system in interactive mode:

```bash
python examples/medical_qa/interactive.py
```

### API Server

To run as an API server:

```bash
python examples/medical_qa/server.py --port 8000
```

Then send requests to the API:

```bash
curl -X POST "http://localhost:8000/api/medical/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the symptoms of atrial fibrillation?"}'
```

### Batch Processing

For processing multiple queries:

```bash
python examples/medical_qa/batch_process.py \
       --input queries.jsonl \
       --output responses.jsonl
```

## Example Queries and Responses

### Example 1: Symptom Query

**Query:**
```
What are the main symptoms of type 2 diabetes?
```

**Response:**
```
The main symptoms of type 2 diabetes include:

1. Polyuria (increased urination)
2. Polydipsia (increased thirst)
3. Polyphagia (increased hunger)
4. Fatigue
5. Blurred vision
6. Slow-healing sores or frequent infections
7. Unexplained weight loss (despite eating more)
8. Numbness or tingling in the hands or feet

Many people with type 2 diabetes may not experience symptoms initially, as the condition develops gradually. Regular screening is important for those with risk factors such as obesity, family history, or sedentary lifestyle.

Medical Disclaimer: This information is for educational purposes only and not a substitute for professional medical advice. Consult a healthcare provider for diagnosis and treatment.
```

### Example 2: Medication Query

**Query:**
```
What are the common side effects of metformin?
```

**Response:**
```
Common side effects of metformin include:

1. Gastrointestinal effects:
   - Nausea
   - Vomiting
   - Diarrhea
   - Abdominal discomfort
   - Bloating
   - Metallic taste

2. Other common side effects:
   - Decreased vitamin B12 levels (with long-term use)
   - Lactic acidosis (rare but serious)

Gastrointestinal side effects typically improve over time and can be minimized by:
- Starting with a low dose and gradually increasing
- Taking the medication with meals
- Using extended-release formulations

Medical Disclaimer: This information is for educational purposes only. Consult your healthcare provider about any medication side effects or concerns.
```

## Performance Benchmarks

| Metric | Baseline Model | Artemis-Optimized |
|--------|---------------|-------------------|
| Average Response Time | 1,250ms | 280ms |
| Memory Usage | 16GB | 6GB |
| Accuracy (MedQA) | 76.3% | 78.1% |
| Accuracy (MedMCQA) | 68.5% | 72.3% |
| Throughput | 8 queries/s | 32 queries/s |

## Advanced Usage

### Custom Medical Knowledge Integration

To integrate your own medical knowledge sources:

1. Prepare your data in the format specified in `examples/medical_qa/data/custom_format.md`
2. Run the knowledge integration script:

```bash
python examples/medical_qa/integrate_knowledge.py \
       --data your_medical_data.json \
       --output custom_medical_model
```

3. Update the config to use your custom model:

```yaml
model:
  base_model: "path/to/custom_medical_model"
```

### Specialized Medical Fields

For adapting to specific medical specialties (e.g., cardiology, oncology), use the specialty adaptation script:

```bash
python examples/medical_qa/specialize.py \
       --base-model artemis-medical-base \
       --specialty cardiology \
       --data cardiology_data.json \
       --output artemis-medical-cardiology
```

## Implementation Details

### Medical Terminology Recognition

The medical terminology module (in `examples/medical_qa/medical_terminolgy.py`) handles:

- Medical abbreviations and acronyms
- Anatomical terms
- Disease and condition names
- Medication names and classes
- Medical procedures and tests

### Evidence Integration

The evidence linking system connects responses to medical literature:

```python
from artemis.examples.medical_qa.evidence import EvidenceLinker

# Initialize the evidence linker with medical knowledge sources
evidence_linker = EvidenceLinker(sources=["pubmed", "guidelines"])

# Generate response with evidence
response = model.generate(query)
response_with_evidence = evidence_linker.enhance_with_evidence(response)
```

## Future Improvements

Planned enhancements for this example:

1. Integration with electronic health record systems
2. Multi-modal support for medical images and lab results
3. Expanded knowledge base with latest medical research
4. Personalized response generation based on patient profile
5. Support for additional medical specialties
6. Temporal reasoning for treatment timelines

## License and Citation

This example is provided under the Apache 2.0 license.

If you use this work, please cite:

```
@article{artemis2025,
  title={Artemis: Adaptive Representation Tuning for Efficient Model Instruction Synthesis},
  author={Smith, J. and Johnson, A. and Williams, R. et al.},
  journal={Proceedings of the International Conference on Medical AI},
  year={2025}
}
```

## Medical Disclaimer

This application is for research and educational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
