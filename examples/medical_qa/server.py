#!/usr/bin/env python
"""
API server for the Medical QA system.
Provides REST endpoints for medical question answering.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# Add parent directory to path to allow importing from artemis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import FastAPI for API server
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Error: Required packages not found. Please install them with:")
    print("pip install fastapi uvicorn pydantic")
    sys.exit(1)

from examples.medical_qa.model import MedicalQAModel, load_medical_qa_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API Models
class MedicalQuery(BaseModel):
    """Request model for medical query."""
    query: str = Field(..., description="Medical question", min_length=3)
    include_evidence: bool = Field(False, description="Include evidence links")
    include_entities: bool = Field(False, description="Extract medical entities")

class BatchMedicalQuery(BaseModel):
    """Request model for batch medical queries."""
    queries: List[str] = Field(..., description="List of medical questions")
    include_evidence: bool = Field(False, description="Include evidence links")
    include_entities: bool = Field(False, description="Extract medical entities")

class MedicalResponse(BaseModel):
    """Response model for medical query."""
    response: str = Field(..., description="Response to medical question")
    entities: Optional[List[Dict]] = Field(None, description="Extracted medical entities")
    evidence: Optional[List[Dict]] = Field(None, description="Evidence citations")
    latency_ms: float = Field(..., description="Processing time in milliseconds")

class BatchMedicalResponse(BaseModel):
    """Response model for batch medical queries."""
    responses: List[MedicalResponse] = Field(..., description="List of responses")
    total_latency_ms: float = Field(..., description="Total processing time in milliseconds")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model information")
    version: str = Field(..., description="API version")

class ErrorResponse(BaseModel):
    """Response model for error."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")

# Globals
app = FastAPI(
    title="Artemis Medical QA API",
    description="API for medical question answering using the Artemis framework",
    version="1.0.0",
)
model = None  # Will be initialized in startup

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on server startup."""
    global model
    
    # Get configuration from command line args
    parser = argparse.ArgumentParser(description="Medical QA API Server")
    parser.add_argument(
        "--config", 
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        help="Path to pre-trained model (overrides config)"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host", 
        type=str,
        default="0.0.0.0",
        help="Host to run the server on"
    )
    args, _ = parser.parse_known_args()
    
    # Initialize the model
    logger.info("Initializing Medical QA model...")
    try:
        if args.model_path:
            model = load_medical_qa_model(args.model_path)
        else:
            model = MedicalQAModel(args.config)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        # We'll let the server start, but the /health endpoint will report error

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "status": "healthy",
        "model": model.config["model"]["base_model"],
        "version": "1.0.0"
    }

@app.post("/api/medical/query", response_model=MedicalResponse, tags=["Medical QA"])
async def process_query(query_data: MedicalQuery):
    """
    Process a medical question and return a response.
    
    Parameters:
    - query: Medical question
    - include_evidence: Whether to include evidence links
    - include_entities: Whether to extract medical entities
    
    Returns:
    - Medical response with optional entities and evidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    
    try:
        # Extract medical entities if requested
        entities = None
        if query_data.include_entities and hasattr(model, 'entity_extractor'):
            entities = model.entity_extractor.extract(query_data.query)
        
        # Generate response
        response = model.generate(query_data.query)
        
        # Get evidence if requested
        evidence = None
        if query_data.include_evidence and hasattr(model, 'evidence_linker'):
            evidence = model.evidence_linker.get_evidence(response, query_data.query)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "response": response,
            "entities": entities,
            "evidence": evidence,
            "latency_ms": latency_ms
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/medical/batch", response_model=BatchMedicalResponse, tags=["Medical QA"])
async def process_batch(batch_data: BatchMedicalQuery):
    """
    Process a batch of medical questions.
    
    Parameters:
    - queries: List of medical questions
    - include_evidence: Whether to include evidence links
    - include_entities: Whether to extract medical entities
    
    Returns:
    - List of medical responses
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    responses = []
    
    try:
        # Process queries
        batch_responses = model.batch_generate(batch_data.queries)
        
        # Process each response
        for i, (query, response) in enumerate(zip(batch_data.queries, batch_responses)):
            item_start_time = time.time()
            
            # Extract entities if requested
            entities = None
            if batch_data.include_entities and hasattr(model, 'entity_extractor'):
                entities = model.entity_extractor.extract(query)
            
            # Get evidence if requested
            evidence = None
            if batch_data.include_evidence and hasattr(model, 'evidence_linker'):
                evidence = model.evidence_linker.get_evidence(response, query)
            
            # Calculate item latency
            item_latency_ms = (time.time() - item_start_time) * 1000
            
            responses.append({
                "response": response,
                "entities": entities,
                "evidence": evidence,
                "latency_ms": item_latency_ms
            })
        
        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "responses": responses,
            "total_latency_ms": total_latency_ms
        }
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/medical/stats", tags=["System"])
async def get_stats():
    """Get system statistics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    import torch
    
    stats = {
        "model_info": {
            "base_model": model.config["model"]["base_model"],
            "optimizations": []
        },
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "device": str(model.device),
            "memory_allocated": None,
            "memory_reserved": None
        }
    }
    
    # Add optimization info
    if model.config['hybrid_adapter']['enabled']:
        stats["model_info"]["optimizations"].append({
            "type": "hybrid_adapter",
            "lora_rank": model.config['hybrid_adapter']['lora_rank'],
            "adapter_size": model.config['hybrid_adapter']['adapter_size']
        })
    
    if model.config['pruning']['enabled']:
        stats["model_info"]["optimizations"].append({
            "type": "pruning",
            "method": model.config['pruning']['method'],
            "sparsity": model.config['pruning']['sparsity']
        })
    
    if model.config['efficiency_transformer']['enabled']:
        stats["model_info"]["optimizations"].append({
            "type": "efficiency_transformer",
            "attention_method": model.config['efficiency_transformer']['attention_method']
        })
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        stats["system_info"]["memory_allocated"] = float(torch.cuda.memory_allocated() / 1024**2)  # MB
        stats["system_info"]["memory_reserved"] = float(torch.cuda.memory_reserved() / 1024**2)  # MB
    
    return stats

# For running the server directly
def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="Medical QA API Server")
    parser.add_argument(
        "--config", 
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        help="Path to pre-trained model (overrides config)"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host", 
        type=str,
        default="0.0.0.0",
        help="Host to run the server on"
    )
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()
