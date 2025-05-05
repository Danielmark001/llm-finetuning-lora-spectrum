#!/usr/bin/env python
"""
Interactive command-line interface for the Multilingual Customer Support system.
"""

import argparse
import logging
import os
import sys
import random
import uuid
from pathlib import Path

# Add parent directory to path to allow importing from artemis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples.multilingual_support.model import MultilingualSupportModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Display configuration
DISPLAY_WIDTH = 80
DEFAULT_LANGUAGE = "en"

# Multilingual help phrases
HELP_PHRASES = {
    "en": "help",
    "es": "ayuda",
    "fr": "aide",
    "de": "hilfe",
    "it": "aiuto",
    "pt": "ajuda",
    "ja": "ヘルプ",
    "zh": "帮助",
    "ko": "도움말",
    "ru": "помощь",
    "ar": "مساعدة",
    "hi": "मदद"
}

# Multilingual welcome messages
WELCOME_MESSAGES = {
    "en": "Welcome to the Artemis Multilingual Customer Support System",
    "es": "Bienvenido al Sistema de Atención al Cliente Multilingüe Artemis",
    "fr": "Bienvenue sur le Système d'Assistance Clientèle Multilingue Artemis",
    "de": "Willkommen beim Artemis Mehrsprachigen Kundendienstsystem",
    "it": "Benvenuto al Sistema di Supporto Clienti Multilingue Artemis",
    "pt": "Bem-vindo ao Sistema de Suporte ao Cliente Multilíngue Artemis",
    "ja": "Artemis多言語カスタマーサポートシステムへようこそ",
    "zh": "欢迎使用Artemis多语言客户支持系统",
    "ko": "Artemis 다국어 고객 지원 시스템에 오신 것을 환영합니다",
    "ru": "Добро пожаловать в Многоязычную Систему Поддержки Клиентов Artemis",
    "ar": "مرحبًا بكم في نظام دعم العملاء متعدد اللغات من Artemis",
    "hi": "आर्टेमिस बहुभाषी ग्राहक सहायता प्रणाली में आपका स्वागत है"
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive Multilingual Customer Support System")
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
        "--language", 
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Initial language code (e.g., 'en', 'es', 'fr')"
    )
    parser.add_argument(
        "--session_id",
        type=str,
        help="Session ID for conversation context (random UUID if not provided)"
    )
    parser.add_argument(
        "--log_level", 
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    return parser.parse_args()

def print_welcome_message(language="en"):
    """Print welcome message and instructions in the specified language."""
    welcome_message = WELCOME_MESSAGES.get(language, WELCOME_MESSAGES["en"])
    
    print("\n" + "=" * DISPLAY_WIDTH)
    print(welcome_message.center(DISPLAY_WIDTH))
    print("=" * DISPLAY_WIDTH)
    
    # Always print instructions in the selected language AND English for clarity
    if language != "en":
        print(f"\nType '{HELP_PHRASES.get(language, 'help')}' for available commands in {language}.")
        print("Type 'help' for available commands in English.")
    else:
        print("\nType 'help' for available commands.")
    
    print("Type 'exit' or 'quit' to end the session.")
    print("The system automatically detects language changes.")
    print("-" * DISPLAY_WIDTH)

def print_help(language="en"):
    """Print available commands in the specified language."""
    if language == "en":
        print("\nAvailable commands:")
        print("  help              - Show this help message")
        print("  exit, quit        - Exit the program")
        print("  clear             - Clear the screen")
        print("  language [code]   - Switch to specified language")
        print("  languages         - List available languages")
        print("  info              - Show model information")
        print("  intent            - Show detected intent of last query")
        print("  sentiment         - Show sentiment analysis of last query")
        print("  examples          - Show example customer queries")
        print("")
    elif language == "es":
        print("\nComandos disponibles:")
        print("  ayuda             - Mostrar este mensaje de ayuda")
        print("  salir, cerrar     - Salir del programa")
        print("  limpiar           - Limpiar la pantalla")
        print("  idioma [código]   - Cambiar al idioma especificado")
        print("  idiomas           - Listar idiomas disponibles")
        print("  info              - Mostrar información del modelo")
        print("  intención         - Mostrar la intención detectada en la última consulta")
        print("  sentimiento       - Mostrar análisis de sentimiento de la última consulta")
        print("  ejemplos          - Mostrar ejemplos de consultas de clientes")
        print("")
    elif language == "fr":
        print("\nCommandes disponibles:")
        print("  aide              - Afficher ce message d'aide")
        print("  sortir, quitter   - Quitter le programme")
        print("  effacer           - Effacer l'écran")
        print("  langue [code]     - Changer de langue")
        print("  langues           - Lister les langues disponibles")
        print("  info              - Afficher les informations du modèle")
        print("  intention         - Afficher l'intention détectée de la dernière requête")
        print("  sentiment         - Afficher l'analyse de sentiment de la dernière requête")
        print("  exemples          - Afficher des exemples de questions clients")
        print("")
    else:
        # For other languages, default to English
        print("\nCommands available (English):")
        print("  help              - Show this help message")
        print("  exit, quit        - Exit the program")
        print("  clear             - Clear the screen")
        print("  language [code]   - Switch to specified language")
        print("  languages         - List available languages")
        print("  info              - Show model information")
        print("  intent            - Show detected intent of last query")
        print("  sentiment         - Show sentiment analysis of last query")
        print("  examples          - Show example customer queries")
        print("")

def show_examples(language="en"):
    """Show example customer queries in the specified language."""
    if language == "en":
        print("\nExample customer support queries:")
        print("  - How do I change my shipping address for an order?")
        print("  - I need to return a defective product.")
        print("  - When will my order arrive?")
        print("  - How do I reset my password?")
        print("  - Can I change my payment method after placing an order?")
        print("  - I'd like to cancel my subscription.")
        print("")
    elif language == "es":
        print("\nEjemplos de consultas de atención al cliente:")
        print("  - ¿Cómo cambio la dirección de envío de un pedido?")
        print("  - Necesito devolver un producto defectuoso.")
        print("  - ¿Cuándo llegará mi pedido?")
        print("  - ¿Cómo restablezco mi contraseña?")
        print("  - ¿Puedo cambiar mi método de pago después de realizar un pedido?")
        print("  - Me gustaría cancelar mi suscripción.")
        print("")
    elif language == "fr":
        print("\nExemples de demandes de support client:")
        print("  - Comment puis-je changer l'adresse de livraison d'une commande?")
        print("  - J'ai besoin de retourner un produit défectueux.")
        print("  - Quand ma commande arrivera-t-elle?")
        print("  - Comment réinitialiser mon mot de passe?")
        print("  - Puis-je changer mon mode de paiement après avoir passé une commande?")
        print("  - Je souhaite annuler mon abonnement.")
        print("")
    else:
        # For other languages, show English examples
        print("\nExample customer support queries (English):")
        print("  - How do I change my shipping address for an order?")
        print("  - I need to return a defective product.")
        print("  - When will my order arrive?")
        print("  - How do I reset my password?")
        print("  - Can I change my payment method after placing an order?")
        print("  - I'd like to cancel my subscription.")
        print("")

def show_model_info(model):
    """Display information about the model."""
    config = model.config
    
    print("\nModel Information:")
    print(f"  Base Model: {config['model']['base_model']}")
    
    # Show supported languages
    print("  Supported Languages:")
    for lang in config['languages']:
        is_default = " (default)" if lang.get('is_default') else ""
        print(f"    - {lang['name']} ({lang['code']}){is_default}")
    
    # Show optimizations
    optimizations = []
    if config['hybrid_adapter']['enabled']:
        opt_type = f"Hybrid Adapter (LoRA rank: {config['hybrid_adapter']['lora_rank']})"
        optimizations.append(opt_type)
    if config['efficiency_transformer']['enabled']:
        opt_type = f"Efficiency Transformer ({config['efficiency_transformer']['attention_method']})"
        optimizations.append(opt_type)
    if config['optimization'].get('quantization'):
        opt_type = f"Quantization ({config['optimization']['quantization']})"
        optimizations.append(opt_type)
    
    print("  Applied Optimizations:")
    for opt in optimizations:
        print(f"    - {opt}")
    
    # Support extensions
    print("  Support Extensions:")
    for ext, enabled in config['support_extensions'].items():
        if isinstance(enabled, bool) and enabled:
            print(f"    - {ext.replace('enable_', '').replace('_', ' ').title()}")
    
    print()

def show_languages(model):
    """Show available languages."""
    print("\nAvailable Languages:")
    for lang in model.config['languages']:
        is_default = " (default)" if lang.get('is_default') else ""
        print(f"  {lang['code']} - {lang['name']}{is_default}")
    print()

def show_intent(intent):
    """Show the detected intent."""
    if intent:
        print(f"\nDetected intent: {intent}")
    else:
        print("\nNo intent detected in the last query.")
    print()

def show_sentiment(sentiment):
    """Show sentiment analysis results."""
    if sentiment:
        print("\nSentiment Analysis:")
        for key, value in sentiment.items():
            print(f"  {key}: {value:.2f}")
    else:
        print("\nNo sentiment analysis available for the last query.")
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
        logger.info("Initializing Multilingual Support model...")
        if args.model_path:
            from examples.multilingual_support.model import load_multilingual_support_model
            model = load_multilingual_support_model(args.model_path)
        else:
            model = MultilingualSupportModel(args.config)
        
        logger.info("Model initialization complete")
        
        # Initialize session
        session_id = args.session_id or str(uuid.uuid4())
        current_language = args.language
        if current_language not in [lang['code'] for lang in model.config['languages']]:
            logger.warning(f"Language {current_language} not supported, using default")
            current_language = model.default_language
        
        # Initialize tracking for previous query
        last_intent = None
        last_sentiment = None
        
        print_welcome_message(current_language)
        
        # Interactive loop
        while True:
            try:
                if current_language in ["ar", "he"]:
                    # For right-to-left languages
                    user_input = input("\n< ").strip()
                else:
                    user_input = input("\n> ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Check for commands (in current language or English)
                if user_input.lower() in ("exit", "quit", "salir", "cerrar", "sortir", "quitter"):
                    print("Exiting Multilingual Support system. Goodbye!")
                    break
                    
                elif user_input.lower() in HELP_PHRASES.values():
                    print_help(current_language)
                    
                elif user_input.lower() in ("clear", "limpiar", "effacer"):
                    clear_screen()
                    print_welcome_message(current_language)
                    
                elif user_input.lower().startswith(("language ", "idioma ", "langue ")):
                    # Handle language switching
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        new_lang = parts[1].strip().lower()
                        if new_lang in [lang['code'] for lang in model.config['languages']]:
                            current_language = new_lang
                            print(f"Language switched to {model.languages[current_language]['name']} ({current_language})")
                        else:
                            print(f"Language {new_lang} not supported")
                            show_languages(model)
                    else:
                        print(f"Current language: {model.languages[current_language]['name']} ({current_language})")
                        
                elif user_input.lower() in ("languages", "idiomas", "langues"):
                    show_languages(model)
                    
                elif user_input.lower() in ("info", "información", "information"):
                    show_model_info(model)
                    
                elif user_input.lower() in ("intent", "intención", "intention"):
                    show_intent(last_intent)
                    
                elif user_input.lower() in ("sentiment", "sentimiento", "sentiment"):
                    show_sentiment(last_sentiment)
                    
                elif user_input.lower() in ("examples", "ejemplos", "exemples"):
                    show_examples(current_language)
                    
                else:
                    # Process customer query
                    print("\nProcessing your query...\n")
                    
                    # Generate response
                    result = model.generate(user_input, language=current_language, session_id=session_id)
                    
                    # Extract and store result data
                    response = result["response"]
                    detected_language = result["language"]
                    
                    # If language was auto-detected and different from current, update it
                    if detected_language != current_language:
                        print(f"Detected language: {model.languages[detected_language]['name']} ({detected_language})")
                        current_language = detected_language
                    
                    # Get and store intent if available
                    if hasattr(model, 'intent_detector'):
                        last_intent, _ = model.intent_detector.detect(user_input, current_language)
                    
                    # Get and store sentiment if available
                    if hasattr(model, 'sentiment_analyzer'):
                        last_sentiment = model.sentiment_analyzer.analyze(user_input, current_language)
                    
                    # Print response
                    print(response)
            
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit or continue with a new question.")
            except EOFError:
                print("\nExiting Multilingual Support system. Goodbye!")
                break
    
    except Exception as e:
        logger.error(f"Error in Multilingual Support system: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
