#!/usr/bin/env python
"""
API server for the Multilingual Customer Support system.
Provides REST endpoints for multilingual customer support.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Add parent directory to path to allow importing from artemis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import FastAPI for API server
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Error: Required packages not found. Please install them with:")
    print("pip install fastapi uvicorn pydantic")
    sys.exit(1)

from examples.multilingual_support.model import MultilingualSupportModel, load_multilingual_support_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API Models
class CustomerQuery(BaseModel):
    """Request model for customer support query."""
    message: str = Field(..., description="Customer query", min_length=2)
    language: Optional[str] = Field(None, description="Language code (auto-detected if not provided)")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation context")
    detect_intent: bool = Field(True, description="Enable intent detection")
    analyze_sentiment: bool = Field(False, description="Enable sentiment analysis")

class BatchCustomerQuery(BaseModel):
    """Request model for batch customer support queries."""
    messages: List[str] = Field(..., description="List of customer queries")
    languages: Optional[List[str]] = Field(None, description="List of language codes (auto-detected if not provided)")
    user_ids: Optional[List[str]] = Field(None, description="List of user identifiers")
    session_ids: Optional[List[str]] = Field(None, description="List of session identifiers")
    detect_intent: bool = Field(True, description="Enable intent detection")
    analyze_sentiment: bool = Field(False, description="Enable sentiment analysis")

class LanguageDetectionRequest(BaseModel):
    """Request model for language detection."""
    text: str = Field(..., description="Text to detect language")

class SupportResponse(BaseModel):
    """Response model for customer support query."""
    response: str = Field(..., description="Response to customer query")
    language: str = Field(..., description="Detected or specified language code")
    detected_intent: Optional[str] = Field(None, description="Detected customer intent")
    sentiment: Optional[Dict[str, float]] = Field(None, description="Sentiment analysis results")
    used_template: bool = Field(False, description="Whether a response template was used")
    latency_ms: float = Field(..., description="Processing time in milliseconds")

class BatchSupportResponse(BaseModel):
    """Response model for batch customer support queries."""
    responses: List[SupportResponse] = Field(..., description="List of responses")
    total_latency_ms: float = Field(..., description="Total processing time in milliseconds")

class LanguageDetectionResponse(BaseModel):
    """Response model for language detection."""
    language: str = Field(..., description="Detected language code")
    language_name: str = Field(..., description="Detected language name")
    confidence: float = Field(..., description="Detection confidence score")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model information")
    version: str = Field(..., description="API version")
    supported_languages: List[Dict[str, str]] = Field(..., description="Supported languages")

class ErrorResponse(BaseModel):
    """Response model for error."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")

# Globals
app = FastAPI(
    title="Artemis Multilingual Customer Support API",
    description="API for multilingual customer support using the Artemis framework",
    version="1.0.0",
)
model = None  # Will be initialized in startup
session_manager = {}  # Simple in-memory session store

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
    parser = argparse.ArgumentParser(description="Multilingual Support API Server")
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
    logger.info("Initializing Multilingual Support model...")
    try:
        if args.model_path:
            model = load_multilingual_support_model(args.model_path)
        else:
            model = MultilingualSupportModel(args.config)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        # We'll let the server start, but the /health endpoint will report error

def get_session_data(session_id: str) -> Dict[str, Any]:
    """Retrieve session data for a given session ID."""
    if session_id not in session_manager:
        session_manager[session_id] = {
            "created_at": time.time(),
            "last_updated": time.time(),
            "language": None,
            "user_id": None,
            "message_count": 0,
            "intents": []
        }
    return session_manager[session_id]

def update_session_data(session_id: str, data: Dict[str, Any]):
    """Update session data for a given session ID."""
    if session_id in session_manager:
        session_manager[session_id].update(data)
        session_manager[session_id]["last_updated"] = time.time()
    else:
        data["created_at"] = time.time()
        data["last_updated"] = time.time()
        session_manager[session_id] = data

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Collect supported languages
    supported_languages = [
        {"code": lang["code"], "name": lang["name"]} 
        for lang in model.config["languages"]
    ]
    
    return {
        "status": "healthy",
        "model": model.config["model"]["base_model"],
        "version": "1.0.0",
        "supported_languages": supported_languages
    }

@app.post("/api/chat", response_model=SupportResponse, tags=["Customer Support"])
async def process_chat(query_data: CustomerQuery):
    """
    Process a customer support query and return a response.
    
    Parameters:
    - message: Customer query text
    - language: Optional language code (auto-detected if not provided)
    - user_id: Optional user identifier
    - session_id: Optional session identifier for conversation context
    - detect_intent: Whether to enable intent detection
    - analyze_sentiment: Whether to enable sentiment analysis
    
    Returns:
    - Support response with detected language and optional metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    
    try:
        # Update session if provided
        if query_data.session_id:
            session_data = get_session_data(query_data.session_id)
            if query_data.user_id:
                session_data["user_id"] = query_data.user_id
            session_data["message_count"] += 1
            if query_data.language:
                session_data["language"] = query_data.language
            update_session_data(query_data.session_id, session_data)
        
        # Generate response
        result = model.generate(
            query_data.message,
            language=query_data.language,
            session_id=query_data.session_id
        )
        
        # Extract data from result
        response = result["response"]
        detected_language = result["language"]
        used_template = result.get("used_template", False)
        
        # Get intent if requested and available
        detected_intent = None
        if query_data.detect_intent and hasattr(model, 'intent_detector'):
            intent, _ = model.intent_detector.detect(
                query_data.message, 
                detected_language
            )
            detected_intent = intent
            
            # Update session with intent
            if query_data.session_id and intent:
                session_data = get_session_data(query_data.session_id)
                session_data["intents"].append(intent)
                update_session_data(query_data.session_id, session_data)
        
        # Get sentiment if requested and available
        sentiment = None
        if query_data.analyze_sentiment and hasattr(model, 'sentiment_analyzer'):
            sentiment = model.sentiment_analyzer.analyze(
                query_data.message,
                detected_language
            )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "response": response,
            "language": detected_language,
            "detected_intent": detected_intent,
            "sentiment": sentiment,
            "used_template": used_template,
            "latency_ms": latency_ms
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch", response_model=BatchSupportResponse, tags=["Customer Support"])
async def process_batch(batch_data: BatchCustomerQuery):
    """
    Process a batch of customer support queries.
    
    Parameters:
    - messages: List of customer queries
    - languages: Optional list of language codes (auto-detected if not provided)
    - user_ids: Optional list of user identifiers
    - session_ids: Optional list of session identifiers
    - detect_intent: Whether to enable intent detection
    - analyze_sentiment: Whether to enable sentiment analysis
    
    Returns:
    - List of support responses
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    responses = []
    
    try:
        # Ensure lists have consistent lengths
        n_messages = len(batch_data.messages)
        languages = batch_data.languages or [None] * n_messages
        user_ids = batch_data.user_ids or [None] * n_messages
        session_ids = batch_data.session_ids or [None] * n_messages
        
        if len(languages) != n_messages or len(session_ids) != n_messages or len(user_ids) != n_messages:
            raise HTTPException(
                status_code=400, 
                detail="If provided, languages, user_ids, and session_ids must have the same length as messages"
            )
        
        # Update sessions if provided
        for i, (message, language, user_id, session_id) in enumerate(
            zip(batch_data.messages, languages, user_ids, session_ids)
        ):
            if session_id:
                session_data = get_session_data(session_id)
                if user_id:
                    session_data["user_id"] = user_id
                session_data["message_count"] += 1
                if language:
                    session_data["language"] = language
                update_session_data(session_id, session_data)
        
        # Generate batch responses
        batch_results = model.batch_generate(
            batch_data.messages,
            languages=languages,
            session_ids=session_ids
        )
        
        # Process each result
        for i, (message, language, user_id, session_id, result) in enumerate(
            zip(batch_data.messages, languages, user_ids, session_ids, batch_results)
        ):
            item_start_time = time.time()
            
            # Extract data from result
            response = result["response"]
            detected_language = result["language"]
            used_template = result.get("used_template", False)
            
            # Get intent if requested and available
            detected_intent = None
            if batch_data.detect_intent and hasattr(model, 'intent_detector'):
                intent, _ = model.intent_detector.detect(message, detected_language)
                detected_intent = intent
                
                # Update session with intent
                if session_id and intent:
                    session_data = get_session_data(session_id)
                    session_data["intents"].append(intent)
                    update_session_data(session_id, session_data)
            
            # Get sentiment if requested and available
            sentiment = None
            if batch_data.analyze_sentiment and hasattr(model, 'sentiment_analyzer'):
                sentiment = model.sentiment_analyzer.analyze(message, detected_language)
            
            # Calculate item latency
            item_latency_ms = (time.time() - item_start_time) * 1000
            
            responses.append({
                "response": response,
                "language": detected_language,
                "detected_intent": detected_intent,
                "sentiment": sentiment,
                "used_template": used_template,
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

@app.post("/api/detect-language", response_model=LanguageDetectionResponse, tags=["Language"])
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of the input text.
    
    Parameters:
    - text: Text to detect language
    
    Returns:
    - Detected language code, name, and confidence score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        detected_lang = model.detect_language(request.text)
        language_name = model.languages[detected_lang]["name"]
        
        # In a real implementation, we would have confidence scores
        # Here we just set a high confidence for simplicity
        confidence = 0.95
        
        return {
            "language": detected_lang,
            "language_name": language_name,
            "confidence": confidence
        }
    
    except Exception as e:
        logger.error(f"Error detecting language: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", tags=["System"])
async def get_stats():
    """Get system statistics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    import torch
    
    stats = {
        "model_info": {
            "base_model": model.config["model"]["base_model"],
            "supported_languages": [lang["code"] for lang in model.config["languages"]],
            "optimizations": []
        },
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "device": str(model.device),
            "memory_allocated": None,
            "memory_reserved": None
        },
        "session_info": {
            "active_sessions": len(session_manager),
            "languages_distribution": {}
        }
    }
    
    # Add optimization info
    if model.config['hybrid_adapter']['enabled']:
        stats["model_info"]["optimizations"].append({
            "type": "hybrid_adapter",
            "lora_rank": model.config['hybrid_adapter']['lora_rank'],
            "adapter_size": model.config['hybrid_adapter']['adapter_size']
        })
    
    if model.config['efficiency_transformer']['enabled']:
        stats["model_info"]["optimizations"].append({
            "type": "efficiency_transformer",
            "attention_method": model.config['efficiency_transformer']['attention_method']
        })
    
    if model.config['optimization'].get('quantization'):
        stats["model_info"]["optimizations"].append({
            "type": "quantization",
            "method": model.config['optimization']['quantization']
        })
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        stats["system_info"]["memory_allocated"] = float(torch.cuda.memory_allocated() / 1024**2)  # MB
        stats["system_info"]["memory_reserved"] = float(torch.cuda.memory_reserved() / 1024**2)  # MB
    
    # Summarize language distribution in active sessions
    language_counts = {}
    for session_id, session_data in session_manager.items():
        lang = session_data.get("language")
        if lang:
            language_counts[lang] = language_counts.get(lang, 0) + 1
    stats["session_info"]["languages_distribution"] = language_counts
    
    return stats

@app.get("/api/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Get session information."""
    if session_id not in session_manager:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return session_manager[session_id]

# For running the server directly
def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="Multilingual Support API Server")
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
