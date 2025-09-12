import subprocess
import time
import json
import os
import re
import argparse
import tiktoken
import statistics
from datetime import datetime, timedelta

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

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

# Parse arguments and load config
args = parse_arguments()
config = load_config(args.config)

if not config:
    print("Failed to load configuration. Exiting.")
    exit(1)

# Apply command line overrides
models = args.models if args.models else config['models']
prompts = config['prompts']
output_file = args.output if args.output else config['output_file']
log_file = config['log_file']
timeout_seconds = args.timeout if args.timeout else config['timeout_seconds']
max_retries = args.retries if args.retries else config['max_retries']

# Set up logging
def log_message(message, print_to_console=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")
    
    if print_to_console:
        print(log_entry)

# If file exists, resume; otherwise start new JSON list
if os.path.exists(output_file):
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        log_message(f"Resuming from existing results file with {len(results)} entries")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log_message(f"Error loading existing results: {e}. Starting fresh.")
        results = []
else:
    results = []
    log_message("Starting new benchmark")

def save_results():
    """Write results to JSON file in real-time"""
    backup_file = output_file + '.backup'
    try:
        # First write to backup file
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # If successful, rename to actual file
        if os.path.exists(output_file):
            os.replace(output_file, output_file + '.old')
        os.rename(backup_file, output_file)
        return True
    except Exception as e:
        log_message(f"Error saving results: {e}")
        return False
    finally:
        # Cleanup backup file if it exists
        if os.path.exists(backup_file):
            os.remove(backup_file)


def estimate_tokens(text):
    """Improved token estimation using common patterns"""
    # Count words (splitting on whitespace)
    word_count = len(text.split())
    
    # Count special tokens (punctuation, numbers, etc.)
    special_tokens = len(re.findall(r'[.,!?;:"\'\(\)\[\]\{\}]', text))
    
    # Count numeric tokens
    numeric_tokens = len(re.findall(r'\d+', text))
    
    # Estimate based on components
    # Generally, each word is ~1.3 tokens, numbers are ~1 token, punctuation is ~1 token
    estimated_tokens = int(word_count * 1.3 + special_tokens + numeric_tokens)
    
    return estimated_tokens

class BenchmarkStats:
    """Track benchmark statistics"""
    def __init__(self, total_tasks):
        self.start_time = time.time()
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.times = []
        self.token_speeds = []
    
    def update(self, elapsed_time, tokens_per_sec):
        self.completed_tasks += 1
        self.times.append(elapsed_time)
        if tokens_per_sec > 0:
            self.token_speeds.append(tokens_per_sec)
    
    def get_progress(self):
        elapsed = time.time() - self.start_time
        if self.completed_tasks > 0:
            avg_time_per_task = elapsed / self.completed_tasks
            remaining_tasks = self.total_tasks - self.completed_tasks
            eta_seconds = avg_time_per_task * remaining_tasks
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."
        
        return {
            "progress": f"{self.completed_tasks}/{self.total_tasks}",
            "percent": round(100 * self.completed_tasks / self.total_tasks, 1),
            "eta": eta,
            "avg_time": statistics.mean(self.times) if self.times else 0,
            "avg_tokens_per_sec": statistics.mean(self.token_speeds) if self.token_speeds else 0
        }

def run_with_retry(model, prompt, max_retries=3, timeout=60):
    """Run ollama command with retry mechanism and timeout"""
    for attempt in range(max_retries):
        try:
            # Run ollama with timeout
            process = subprocess.Popen(
                ["ollama", "run", model, prompt],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace"
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            
            if process.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.strip(),
                    "error": None
                }
            else:
                raise subprocess.CalledProcessError(
                    process.returncode, ["ollama", "run"], stderr.strip()
                )
                
        except subprocess.TimeoutExpired:
            process.kill()
            log_message(f"    ⚠️ Attempt {attempt + 1}: Timeout after {timeout} seconds")
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Timeout after {timeout} seconds"
                }
                
        except subprocess.CalledProcessError as e:
            log_message(f"    ⚠️ Attempt {attempt + 1}: Process error: {e.stderr if e.stderr else str(e)}")
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "output": None,
                    "error": str(e)
                }
                
        except Exception as e:
            log_message(f"    ⚠️ Attempt {attempt + 1}: Unexpected error: {str(e)}")
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "output": None,
                    "error": str(e)
                }
        
        # Wait before retrying, with exponential backoff
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # 1, 2, 4 seconds
            log_message(f"    ⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

# Initialize benchmark stats
total_tasks = len(models) * len(prompts)
stats = BenchmarkStats(total_tasks)

# Loop over models and prompts
for model in models:
    log_message(f"🔍 Testing model: {model}")
    for i, prompt in enumerate(prompts, start=1):
        # Show progress
        progress = stats.get_progress()
        log_message(f"Progress: {progress['progress']} ({progress['percent']}%) - ETA: {progress['eta']}")
        log_message(f"  ➤ Asking question {i}: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
        start = time.time()

        # Run with retry and timeout
        result = run_with_retry(model, prompt)
        
        if result["success"]:
            answer = result["output"]
            token_count = estimate_tokens(answer)
            log_message(f"    ✅ Answer received ({len(answer)} chars, ~{token_count} tokens)")
        else:
            answer = f"Error: {result['error']}"
            token_count = 0
            log_message(f"    ❌ Error: {answer}")

        end = time.time()
        elapsed = round(end - start, 2)
        
        # Calculate tokens per second
        tokens_per_sec = round(token_count / elapsed, 2) if elapsed > 0 and token_count > 0 else 0
        
        # Update statistics
        stats.update(elapsed, tokens_per_sec)
        progress = stats.get_progress()
        
        log_message(f"    ⏱️ Time taken: {elapsed} sec")
        log_message(f"    📊 Speed: {tokens_per_sec} tokens/sec")
        log_message(f"    📈 Running average: {progress['avg_tokens_per_sec']:.2f} tokens/sec")

        # Append result
        entry = {
            "model": model,
            "prompt": prompt,
            "time_taken_sec": elapsed,
            "answer_length": len(answer),
            "estimated_tokens": token_count,
            "tokens_per_sec": tokens_per_sec,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        results.append(entry)

        # Save immediately
        if save_results():
            log_message(f"    💾 Saved result to {output_file}")
        else:
            log_message("    ❌ Failed to save result")

log_message("🏁 Benchmark complete. All results saved to ollama_results.json")