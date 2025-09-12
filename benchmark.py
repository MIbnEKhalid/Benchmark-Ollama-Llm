import subprocess
import time
import json
import os
import re
from datetime import datetime, timedelta
import statistics

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

class OllamaBenchmark:
    def __init__(self, config):
        self.models = config['models']
        self.prompts = config['prompts']
        self.timeout = config.get('timeout_seconds', 60)
        self.max_retries = config.get('max_retries', 3)
        self.output_file = config['output_file']
        self.log_file = config['log_file']
        self.results = self._load_results()
        self.stats = BenchmarkStats(len(self.models) * len(self.prompts))

    def _load_results(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.log_message(f"Error loading existing results: {e}. Starting fresh.")
                return []
        return []

    def log_message(self, message, print_to_console=True):
        """Log a message to file and optionally console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        
        if print_to_console:
            print(log_entry)

    def save_results(self):
        """Write results to JSON file in real-time"""
        backup_file = self.output_file + '.backup'
        try:
            # First write to backup file
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            # If successful, rename to actual file
            if os.path.exists(self.output_file):
                os.replace(self.output_file, self.output_file + '.old')
            os.rename(backup_file, self.output_file)
            return True
        except Exception as e:
            self.log_message(f"Error saving results: {e}")
            return False
        finally:
            if os.path.exists(backup_file):
                os.remove(backup_file)

    def estimate_tokens(self, text):
        """Improved token estimation"""
        word_count = len(text.split())
        special_tokens = len(re.findall(r'[.,!?;:"\'\(\)\[\]\{\}]', text))
        numeric_tokens = len(re.findall(r'\d+', text))
        return int(word_count * 1.3 + special_tokens + numeric_tokens)

    def run_with_retry(self, model, prompt):
        """Run ollama command with retry mechanism and timeout"""
        for attempt in range(self.max_retries):
            try:
                process = subprocess.Popen(
                    ["ollama", "run", model, prompt],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace"
                )
                
                stdout, stderr = process.communicate(timeout=self.timeout)
                
                if process.returncode == 0:
                    return {"success": True, "output": stdout.strip(), "error": None}
                else:
                    raise subprocess.CalledProcessError(
                        process.returncode, ["ollama", "run"], stderr.strip()
                    )
                    
            except subprocess.TimeoutExpired:
                process.kill()
                self.log_message(f"    ⚠️ Attempt {attempt + 1}: Timeout after {self.timeout} seconds")
                if attempt == self.max_retries - 1:
                    return {"success": False, "output": None,
                           "error": f"Timeout after {self.timeout} seconds"}
                    
            except subprocess.CalledProcessError as e:
                self.log_message(f"    ⚠️ Attempt {attempt + 1}: Process error: {e.stderr if e.stderr else str(e)}")
                if attempt == self.max_retries - 1:
                    return {"success": False, "output": None, "error": str(e)}
                    
            except Exception as e:
                self.log_message(f"    ⚠️ Attempt {attempt + 1}: Unexpected error: {str(e)}")
                if attempt == self.max_retries - 1:
                    return {"success": False, "output": None, "error": str(e)}
            
            wait_time = 2 ** attempt
            self.log_message(f"    ⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

    def run_benchmark(self):
        """Run the benchmark"""
        for model in self.models:
            self.log_message(f"🔍 Testing model: {model}")
            for i, prompt in enumerate(self.prompts, start=1):
                # Show progress
                progress = self.stats.get_progress()
                self.log_message(f"Progress: {progress['progress']} ({progress['percent']}%) - ETA: {progress['eta']}")
                self.log_message(f"  ➤ Asking question {i}: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
                
                start = time.time()
                result = self.run_with_retry(model, prompt)
                
                if result["success"]:
                    answer = result["output"]
                    token_count = self.estimate_tokens(answer)
                    self.log_message(f"    ✅ Answer received ({len(answer)} chars, ~{token_count} tokens)")
                else:
                    answer = f"Error: {result['error']}"
                    token_count = 0
                    self.log_message(f"    ❌ Error: {answer}")

                end = time.time()
                elapsed = round(end - start, 2)
                tokens_per_sec = round(token_count / elapsed, 2) if elapsed > 0 and token_count > 0 else 0
                
                # Update statistics
                self.stats.update(elapsed, tokens_per_sec)
                progress = self.stats.get_progress()
                
                self.log_message(f"    ⏱️ Time taken: {elapsed} sec")
                self.log_message(f"    📊 Speed: {tokens_per_sec} tokens/sec")
                self.log_message(f"    📈 Running average: {progress['avg_tokens_per_sec']:.2f} tokens/sec")
                
                # Append result
                entry = {
                    "model": model,
                    "prompt": prompt,
                    "time_taken_sec": elapsed,
                    "answer_length": len(answer),
                    "estimated_tokens": token_count,
                    "tokens_per_sec": tokens_per_sec,                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                }
                self.results.append(entry)

                # Save immediately
                if self.save_results():
                    self.log_message(f"    💾 Saved result to {self.output_file}")
                else:
                    self.log_message("    ❌ Failed to save result")

        self.log_message("🏁 Benchmark complete. All results saved to ollama_results.json")