#!/usr/bin/env python
"""
Interactive command-line interface for the Medical QA system.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing from artemis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples.medical_qa.model import MedicalQAModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive Medical QA System")
    parser.add_argument(
        "--config", 
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model_path", 
        type=str,
        help="Path to pre-trained model (overrides config)"
    )
    parser.add_argument(
        "--log_level", 
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    return parser.parse_args()

def print_welcome_message():
    """Print welcome message and instructions."""
    print("\n" + "=" * 80)
    print("Welcome to the Artemis Medical QA System".center(80))
    print("=" * 80)
    print("\nThis system provides medical information for educational purposes only.")
    print("It is not a substitute for professional medical advice, diagnosis, or treatment.")
    print("\nAsk medical questions or type 'exit', 'quit', or Ctrl+D to end the session.")
    print("Type 'help' for additional commands.")
    print("-" * 80)

def print_help():
    """Print available commands."""
    print("\nAvailable commands:")
    print("  help       - Show this help message")
    print("  exit, quit - Exit the program")
    print("  clear      - Clear the screen")
    print("  info       - Show model information")
    print("  benchmark  - Run a quick performance benchmark")
    print("  examples   - Show example questions")
    print()

def show_examples():
    """Show example medical questions."""
    print("\nExample medical questions you can ask:")
    print("  - What are the symptoms of pneumonia?")
    print("  - How does insulin regulate blood sugar?")
    print("  - What are common side effects of amoxicillin?")
    print("  - What is the difference between an MRI and a CT scan?")
    print("  - How is rheumatoid arthritis diagnosed?")
    print("  - What lifestyle changes can help manage hypertension?")
    print()

def show_model_info(model):
    """Display information about the model."""
    config = model.config
    
    print("\nModel Information:")
    print(f"  Base Model: {config['model']['base_model']}")
    
    # Optimizations
    optimizations = []
    if config['hybrid_adapter']['enabled']:
        opt_type = f"Hybrid Adapter (LoRA rank: {config['hybrid_adapter']['lora_rank']})"
        optimizations.append(opt_type)
    if config['pruning']['enabled']:
        opt_type = f"Pruning ({config['pruning']['method']}, {config['pruning']['sparsity']*100:.1f}% sparsity)"
        optimizations.append(opt_type)
    if config['efficiency_transformer']['enabled']:
        opt_type = f"Efficiency Transformer ({config['efficiency_transformer']['attention_method']})"
        optimizations.append(opt_type)
    
    print("  Applied Optimizations:")
    for opt in optimizations:
        print(f"    - {opt}")
    
    # Medical extensions
    print("  Medical Extensions:")
    for ext, enabled in config['medical_extensions'].items():
        if ext != "knowledge_sources" and enabled:
            print(f"    - {ext.replace('_', ' ').title()}")
    
    print(f"  Knowledge Sources: {', '.join(config['medical_extensions']['knowledge_sources'])}")
    print()

def run_benchmark(model):
    """Run a quick performance benchmark."""
    import time
    import statistics
    
    test_questions = [
        "What are the symptoms of the flu?",
        "How does metformin work to control blood sugar?",
        "What is the difference between Alzheimer's and dementia?",
        "What are the risk factors for heart disease?",
        "How is pneumonia diagnosed?"
    ]
    
    print("\nRunning benchmark with 5 sample questions...")
    latencies = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"  Processing question {i}/5...", end="", flush=True)
        start_time = time.time()
        _ = model.generate(question)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)
        print(f" {latency:.2f}ms")
    
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    
    print("\nBenchmark Results:")
    print(f"  Average Latency: {avg_latency:.2f}ms")
    print(f"  Median Latency: {median_latency:.2f}ms")
    print(f"  Throughput: {1000/avg_latency:.2f} questions/second")
    print()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main interactive loop."""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize the model
        logger.info("Initializing Medical QA model...")
        if args.model_path:
            from examples.medical_qa.model import load_medical_qa_model
            model = load_medical_qa_model(args.model_path)
        else:
            model = MedicalQAModel(args.config)
        
        logger.info("Model initialization complete")
        
        print_welcome_message()
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n> ").strip()
                
                # Check for commands
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting Medical QA system. Goodbye!")
                    break
                elif user_input.lower() == "help":
                    print_help()
                elif user_input.lower() == "clear":
                    clear_screen()
                    print_welcome_message()
                elif user_input.lower() == "info":
                    show_model_info(model)
                elif user_input.lower() == "benchmark":
                    run_benchmark(model)
                elif user_input.lower() == "examples":
                    show_examples()
                elif user_input:
                    # Process medical question
                    print("\nProcessing your question...\n")
                    response = model.generate(user_input)
                    print(response)
            
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit or continue with a new question.")
            except EOFError:
                print("\nExiting Medical QA system. Goodbye!")
                break
    
    except Exception as e:
        logger.error(f"Error in Medical QA system: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
