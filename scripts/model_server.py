#!/usr/bin/env python3
"""
Model Serving Script for LLM Fine-Tuning Project
================================================
This script provides a simple API server for deploying fine-tuned language models.
It supports both REST API and WebSocket interfaces for real-time inference.
"""

import os
import sys
import json
import argparse
import logging
import yaml
import time
import asyncio
import threading
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import server frameworks
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, Request
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: FastAPI and uvicorn are required for the model server.")
    print("Install them with: pip install fastapi uvicorn")
    sys.exit(1)

# Import model serving frameworks
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
        BitsAndBytesConfig,
        GenerationConfig,
    )
    from peft import PeftModel
except ImportError:
    print("Error: Transformers and related libraries are required.")
    print("Install them with: pip install transformers torch peft")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Define API models
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=5.0)
    do_sample: bool = Field(default=True)
    stream: bool = Field(default=False)
    system_prompt: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=5.0)
    do_sample: bool = Field(default=True)
    stream: bool = Field(default=False)


class ModelInfo(BaseModel):
    model_name: str
    base_model: str
    adapter_name: Optional[str] = None
    quantization: Optional[str] = None
    max_context_length: int
    server_version: str = "1.0.0"
    loaded_at: str


# Global variables
model = None
tokenizer = None
model_info = {}
active_generations = 0
max_concurrent_generations = 4
generation_semaphore = None
SERVER_VERSION = "1.0.0"


def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
    use_flash_attention: bool = False,
    device_map: str = "auto",
) -> tuple:
    """
    Load a model for inference with optimizations.
    
    Args:
        model_path: Path to the model or model identifier
        adapter_path: Path to the PEFT adapter
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        use_flash_attention: Whether to use flash attention
        device_map: Device mapping strategy
        
    Returns:
        tuple: (model, tokenizer, model_info)
    """
    global model_info
    
    logger.info(f"Loading model from {model_path}...")
    
    # Quantization config
    quantization_config = None
    quantization_type = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        quantization_type = "4-bit (NF4)"
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        quantization_type = "8-bit"
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": device_map,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # Flash attention configuration
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # Load adapter if provided
    if adapter_path:
        logger.info(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        adapter_name = os.path.basename(adapter_path)
    else:
        adapter_name = None
    
    # Extract model name
    if "/" in model_path:
        model_name = model_path.split("/")[-1]
    else:
        model_name = model_path
    
    # Create model info
    model_info = {
        "model_name": model_name,
        "base_model": model_path,
        "adapter_name": adapter_name,
        "quantization": quantization_type,
        "max_context_length": getattr(model.config, "max_position_embeddings", 4096),
        "server_version": SERVER_VERSION,
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    
    logger.info(f"Model loaded successfully with {model.num_parameters():,} parameters")
    
    return model, tokenizer, model_info


def create_app(
    model_path: str,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
    use_flash_attention: bool = False,
    max_concurrent: int = 4,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        model_path: Path to the model or model identifier
        adapter_path: Path to the PEFT adapter
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        use_flash_attention: Whether to use flash attention
        max_concurrent: Maximum number of concurrent generations
        
    Returns:
        FastAPI: Configured application
    """
    global model, tokenizer, model_info, max_concurrent_generations, generation_semaphore
    
    # Set max concurrent generations
    max_concurrent_generations = max_concurrent
    generation_semaphore = asyncio.Semaphore(max_concurrent_generations)
    
    # Load model
    model, tokenizer, model_info = load_model(
        model_path=model_path,
        adapter_path=adapter_path,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        use_flash_attention=use_flash_attention,
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="LLM Fine-Tuning Model Server",
        description="API server for fine-tuned language models",
        version=SERVER_VERSION,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define API endpoints
    
    @app.get("/")
    async def root():
        return {"status": "ok", "message": "LLM Fine-Tuning Model Server"}
    
    @app.get("/model/info")
    async def get_model_info():
        return model_info
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "active_generations": active_generations,
            "max_concurrent_generations": max_concurrent_generations,
        }
    
    @app.post("/generate")
    async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
        global active_generations
        
        # Check if we're over capacity
        if active_generations >= max_concurrent_generations:
            raise HTTPException(
                status_code=429,
                detail=f"Server at capacity. Maximum concurrent generations: {max_concurrent_generations}",
            )
        
        # If streaming is requested, use streaming endpoint
        if request.stream:
            return StreamingResponse(
                stream_generate(request),
                media_type="text/event-stream",
            )
        
        # Format prompt with system prompt if provided
        formatted_prompt = request.prompt
        if request.system_prompt:
            if hasattr(tokenizer, "apply_chat_template"):
                # Use chat template if available
                messages = [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.prompt},
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fall back to simple format
                formatted_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant: "
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        
        # Tokenize input
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Track active generations
        active_generations += 1
        
        try:
            # Generate text
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    generation_config=gen_config,
                )
            
            # Decode the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if request.system_prompt and hasattr(tokenizer, "apply_chat_template"):
                # For chat template, extract the last assistant message
                assistant_response = generated_text.split("Assistant: ")[-1].strip()
            else:
                # For simple format, take everything after the prompt
                assistant_response = generated_text[len(formatted_prompt):].strip()
            
            # Return the result
            return {
                "generated_text": assistant_response,
                "full_text": generated_text,
            }
        
        finally:
            # Clean up and reduce active generations count
            background_tasks.add_task(reduce_active_generations)
    
    async def stream_generate(request: GenerationRequest):
        global active_generations
        
        async with generation_semaphore:
            active_generations += 1
            try:
                # Format prompt with system prompt if provided
                formatted_prompt = request.prompt
                if request.system_prompt:
                    if hasattr(tokenizer, "apply_chat_template"):
                        # Use chat template if available
                        messages = [
                            {"role": "system", "content": request.system_prompt},
                            {"role": "user", "content": request.prompt},
                        ]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        # Fall back to simple format
                        formatted_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant: "
                
                # Create generation config
                gen_config = GenerationConfig(
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                )
                
                # Tokenize input
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
                
                # Create streamer
                streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
                
                # Start generation in a separate thread
                generation_kwargs = {
                    "input_ids": input_ids,
                    "generation_config": gen_config,
                    "streamer": streamer,
                }
                
                thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Determine the length of the prompt in the generated text
                prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                prompt_len = len(prompt_text)
                
                # Stream the results
                generated_text = ""
                for text in streamer:
                    generated_text += text
                    # Only stream the new part after the prompt
                    if len(generated_text) > prompt_len:
                        chunk = generated_text[prompt_len:]
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                
                yield f"data: {json.dumps({'text': generated_text[prompt_len:], 'done': True})}\n\n"
                yield "data: [DONE]\n\n"
            
            finally:
                active_generations -= 1
    
    @app.post("/chat")
    async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
        global active_generations
        
        # Check if we're over capacity
        if active_generations >= max_concurrent_generations:
            raise HTTPException(
                status_code=429,
                detail=f"Server at capacity. Maximum concurrent generations: {max_concurrent_generations}",
            )
        
        # If streaming is requested, use streaming endpoint
        if request.stream:
            return StreamingResponse(
                stream_chat(request),
                media_type="text/event-stream",
            )
        
        # Convert messages to the format expected by the tokenizer
        messages = []
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fall back to simple format
            formatted_prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted_prompt += f"{role.capitalize()}: {content}\n\n"
            formatted_prompt += "Assistant: "
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        
        # Tokenize input
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Track active generations
        active_generations += 1
        
        try:
            # Generate text
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    generation_config=gen_config,
                )
            
            # Decode the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if hasattr(tokenizer, "apply_chat_template"):
                # For chat template, extract the last assistant message
                assistant_response = generated_text.split("Assistant: ")[-1].strip()
            else:
                # For simple format, take everything after the last "Assistant: "
                assistant_response = generated_text.split("Assistant: ")[-1].strip()
            
            # Create the response message
            response_message = ChatMessage(role="assistant", content=assistant_response)
            
            # Return the result
            return {
                "message": response_message,
                "full_text": generated_text,
            }
        
        finally:
            # Clean up and reduce active generations count
            background_tasks.add_task(reduce_active_generations)
    
    async def stream_chat(request: ChatRequest):
        global active_generations
        
        async with generation_semaphore:
            active_generations += 1
            try:
                # Convert messages to the format expected by the tokenizer
                messages = []
                for msg in request.messages:
                    messages.append({"role": msg.role, "content": msg.content})
                
                # Apply chat template
                if hasattr(tokenizer, "apply_chat_template"):
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    # Fall back to simple format
                    formatted_prompt = ""
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]
                        formatted_prompt += f"{role.capitalize()}: {content}\n\n"
                    formatted_prompt += "Assistant: "
                
                # Create generation config
                gen_config = GenerationConfig(
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                )
                
                # Tokenize input
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
                
                # Create streamer
                streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
                
                # Start generation in a separate thread
                generation_kwargs = {
                    "input_ids": input_ids,
                    "generation_config": gen_config,
                    "streamer": streamer,
                }
                
                thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Extract assistant's prefix in the prompt to determine where to start streaming
                last_assistant_idx = formatted_prompt.rfind("Assistant: ")
                assistant_prefix = ""
                if last_assistant_idx != -1:
                    assistant_prefix = formatted_prompt[last_assistant_idx + len("Assistant: "):]
                
                # Stream the results
                generated_text = ""
                for text in streamer:
                    generated_text += text
                    # Skip the prompt part and only yield new tokens
                    if len(generated_text) > len(formatted_prompt) - len(assistant_prefix):
                        chunk = generated_text[len(formatted_prompt) - len(assistant_prefix):]
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                
                yield f"data: {json.dumps({'text': generated_text[len(formatted_prompt) - len(assistant_prefix):], 'done': True})}\n\n"
                yield "data: [DONE]\n\n"
            
            finally:
                active_generations -= 1
    
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        await websocket.accept()
        
        try:
            while True:
                # Receive request JSON
                data = await websocket.receive_text()
                request_data = json.loads(data)
                
                # Validate request
                try:
                    if "messages" in request_data:
                        request = ChatRequest(**request_data)
                        endpoint = "chat"
                    else:
                        request = GenerationRequest(**request_data)
                        endpoint = "generate"
                except Exception as e:
                    await websocket.send_json({"error": f"Invalid request: {str(e)}"})
                    continue
                
                # Check if we're over capacity
                if active_generations >= max_concurrent_generations:
                    await websocket.send_json({
                        "error": f"Server at capacity. Maximum concurrent generations: {max_concurrent_generations}"
                    })
                    continue
                
                # Process the request based on endpoint
                if endpoint == "chat":
                    await process_ws_chat(websocket, request)
                else:
                    await process_ws_generate(websocket, request)
        
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            await websocket.close()
    
    async def process_ws_generate(websocket: WebSocket, request: GenerationRequest):
        global active_generations
        
        async with generation_semaphore:
            active_generations += 1
            try:
                # Format prompt with system prompt if provided
                formatted_prompt = request.prompt
                if request.system_prompt:
                    if hasattr(tokenizer, "apply_chat_template"):
                        # Use chat template if available
                        messages = [
                            {"role": "system", "content": request.system_prompt},
                            {"role": "user", "content": request.prompt},
                        ]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        # Fall back to simple format
                        formatted_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant: "
                
                # Create generation config
                gen_config = GenerationConfig(
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                )
                
                # Tokenize input
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
                
                # If streaming, stream results
                if request.stream:
                    # Create streamer
                    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
                    
                    # Start generation in a separate thread
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "generation_config": gen_config,
                        "streamer": streamer,
                    }
                    
                    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # Determine where the prompt ends
                    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    prompt_len = len(prompt_text)
                    
                    # Stream results
                    generated_text = ""
                    for text in streamer:
                        generated_text += text
                        # Only stream new content after the prompt
                        if len(generated_text) > prompt_len:
                            chunk = generated_text[prompt_len:]
                            await websocket.send_json({
                                "text": chunk,
                                "done": False,
                            })
                    
                    # Send final message
                    await websocket.send_json({
                        "text": generated_text[prompt_len:],
                        "done": True,
                    })
                
                else:
                    # Generate text without streaming
                    with torch.no_grad():
                        output = model.generate(
                            input_ids,
                            generation_config=gen_config,
                        )
                    
                    # Decode the output
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Extract the assistant's response
                    if request.system_prompt and hasattr(tokenizer, "apply_chat_template"):
                        # For chat template, extract the last assistant message
                        assistant_response = generated_text.split("Assistant: ")[-1].strip()
                    else:
                        # For simple format, take everything after the prompt
                        assistant_response = generated_text[len(formatted_prompt):].strip()
                    
                    # Send result
                    await websocket.send_json({
                        "generated_text": assistant_response,
                        "full_text": generated_text,
                    })
            
            finally:
                active_generations -= 1
    
    async def process_ws_chat(websocket: WebSocket, request: ChatRequest):
        global active_generations
        
        async with generation_semaphore:
            active_generations += 1
            try:
                # Convert messages to the format expected by the tokenizer
                messages = []
                for msg in request.messages:
                    messages.append({"role": msg.role, "content": msg.content})
                
                # Apply chat template
                if hasattr(tokenizer, "apply_chat_template"):
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    # Fall back to simple format
                    formatted_prompt = ""
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]
                        formatted_prompt += f"{role.capitalize()}: {content}\n\n"
                    formatted_prompt += "Assistant: "
                
                # Create generation config
                gen_config = GenerationConfig(
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                )
                
                # Tokenize input
                input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
                
                # If streaming, stream results
                if request.stream:
                    # Create streamer
                    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
                    
                    # Start generation in a separate thread
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "generation_config": gen_config,
                        "streamer": streamer,
                    }
                    
                    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # Extract assistant's prefix in the prompt
                    last_assistant_idx = formatted_prompt.rfind("Assistant: ")
                    assistant_prefix = ""
                    if last_assistant_idx != -1:
                        assistant_prefix = formatted_prompt[last_assistant_idx + len("Assistant: "):]
                    
                    # Stream results
                    generated_text = ""
                    for text in streamer:
                        generated_text += text
                        # Skip the prompt and only send new generated content
                        if len(generated_text) > len(formatted_prompt) - len(assistant_prefix):
                            chunk = generated_text[len(formatted_prompt) - len(assistant_prefix):]
                            await websocket.send_json({
                                "text": chunk,
                                "done": False,
                            })
                    
                    # Send final message
                    await websocket.send_json({
                        "text": generated_text[len(formatted_prompt) - len(assistant_prefix):],
                        "done": True,
                    })
                
                else:
                    # Generate text without streaming
                    with torch.no_grad():
                        output = model.generate(
                            input_ids,
                            generation_config=gen_config,
                        )
                    
                    # Decode the output
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Extract the assistant's response
                    if hasattr(tokenizer, "apply_chat_template"):
                        # For chat template, extract the last assistant message
                        assistant_response = generated_text.split("Assistant: ")[-1].strip()
                    else:
                        # For simple format, take everything after the last "Assistant: "
                        assistant_response = generated_text.split("Assistant: ")[-1].strip()
                    
                    # Create the response message
                    response_message = {"role": "assistant", "content": assistant_response}
                    
                    # Send result
                    await websocket.send_json({
                        "message": response_message,
                        "full_text": generated_text,
                    })
            
            finally:
                active_generations -= 1
    
    def reduce_active_generations():
        """Background task to reduce active generations count."""
        global active_generations
        active_generations -= 1
    
    return app


def main():
    """Main function to handle CLI arguments and start the server."""
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Model Server")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or model identifier")
    parser.add_argument("--adapter_path", type=str, help="Path to the PEFT adapter")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use flash attention")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--max_concurrent", type=int, default=4, help="Maximum number of concurrent generations")
    
    args = parser.parse_args()
    
    # Create and run the app
    app = create_app(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=args.use_flash_attention,
        max_concurrent=args.max_concurrent,
    )
    
    # Run the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
