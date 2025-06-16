import torch
import time
import csv

# Config
N = 4096                         # Matrix dimension (NxN)
dtype = torch.float16            # Use float32 for FP32 tests
num_runs = 31                    # Number of times to repeat the test
csv_filename = "fp16_benchmark_results.csv"

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

# Create matrices once
A = torch.randn(N, N, device=device, dtype=dtype)
B = torch.randn(N, N, device=device, dtype=dtype)

# Warm-up
for _ in range(5):
    torch.matmul(A, B)

# Store results
results = []

# Run benchmark multiple times
for run in range(1, num_runs + 1):
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed_time = end - start
    flops = 2 * (N ** 3)
    tflops = flops / elapsed_time / 1e12

    results.append((run, elapsed_time, tflops))
    print(f"Run {run}: Time = {elapsed_time:.6f}s, TFLOPs = {tflops:.2f}")

# Write to CSV
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run', 'Elapsed Time (s)', 'TFLOPs'])
    writer.writerows(results)

print(f"\nSaved results to '{csv_filename}'")