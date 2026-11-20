import time
import torch

class CallTracker:
    def __init__(self):
        self.wall_start_time = 0.0
        self.wall_end_time = 0.0
        
        # Generator metrics
        self.generator_calls = 0
        self.generator_gpu_time = 0.0
        
        # Verifier metrics
        self.verifier_calls = 0
        self.verifier_gpu_time = 0.0

    def start_wall_timer(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.wall_start_time = time.perf_counter()

    def stop_wall_timer(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.wall_end_time = time.perf_counter()

    def record_call(self, call_type: str, elapsed_gpu_time: float):
        if call_type == "generator":
            self.generator_calls += 1
            self.generator_gpu_time += elapsed_gpu_time
        else:
            self.verifier_calls += 1
            self.verifier_gpu_time += elapsed_gpu_time

    def print_summary(self, run_name: str = "Experiment"):
        total_wall_time = self.wall_end_time - self.wall_start_time
        total_calls = self.generator_calls + self.verifier_calls
        total_gpu_time = self.generator_gpu_time + self.verifier_gpu_time

        print(f"\n==========================================")
        print(f"       SUMMARY: {run_name.upper()}")
        print(f"==========================================")
        print(f" Total Wall-Clock Time : {total_wall_time:.2f} s ({total_wall_time / 60:.2f} min)")
        print(f" Total LLM Calls       : {total_calls}")
        print(f"   ├─ Generator Calls  : {self.generator_calls} ({self.generator_gpu_time:.2f} s GPU time)")
        print(f"   └─ Verifier Calls   : {self.verifier_calls} ({self.verifier_gpu_time:.2f} s GPU time)")
        print(f" Total GPU Gen Time    : {total_gpu_time:.2f} s")
        print(f"==========================================\n")

# Single global instance accessible everywhere
tracker = CallTracker()