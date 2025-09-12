import json
import argparse
from benchmark import OllamaBenchmark

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark Ollama models')
    parser.add_argument('--config', default='benchmark_config.json',
                      help='Path to configuration file')
    parser.add_argument('--models', nargs='+',
                      help='Override models from config file')
    parser.add_argument('--timeout', type=int,
                      help='Override timeout seconds from config file')
    parser.add_argument('--retries', type=int,
                      help='Override max retries from config file')
    parser.add_argument('--output',
                      help='Override output file from config file')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        exit(1)
    
    # Apply command line overrides
    if args.models:
        config['models'] = args.models
    if args.timeout:
        config['timeout_seconds'] = args.timeout
    if args.retries:
        config['max_retries'] = args.retries
    if args.output:
        config['output_file'] = args.output
    
    # Create and run benchmark
    benchmark = OllamaBenchmark(config)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()