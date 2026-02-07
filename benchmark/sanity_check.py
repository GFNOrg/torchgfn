#!/usr/bin/env python

import time

import jax
import jax.numpy as jnp
import torch

# --- Settings ---
# 4096 is a sweet spot: large enough to require GPU power,
# but fits easily in memory.
N = 4096
LOOPS = 50

print(f"--- Benchmark Config: {N}x{N} Matrix Multiplication ---")
print(f"JAX Devices: {jax.devices()}")
print(f"PyTorch Version: {torch.__version__}")

# --- 1. JAX Setup ---
key = jax.random.PRNGKey(0)
jax_x = jax.random.normal(key, (N, N))
jax_y = jax.random.normal(key, (N, N))


def matmul_fn(a, b):
    return jnp.dot(a, b)


jit_matmul = jax.jit(matmul_fn)

# --- 2. JAX EAGER MODE ---
print("\n--- 1. JAX Eager (No JIT) ---")
_ = matmul_fn(jax_x, jax_y).block_until_ready()  # Warmup
start = time.time()
for _ in range(LOOPS):
    _ = matmul_fn(jax_x, jax_y).block_until_ready()
jax_eager_time = (time.time() - start) / LOOPS
print(f"Time: {jax_eager_time:.4f} s")

# --- 3. JAX JIT MODE ---
print("\n--- 2. JAX JIT (Compiled) ---")
print("Compiling...")
_ = jit_matmul(jax_x, jax_y).block_until_ready()  # Compilation triggers here
start = time.time()
for _ in range(LOOPS):
    _ = jit_matmul(jax_x, jax_y).block_until_ready()
jax_jit_time = (time.time() - start) / LOOPS
print(f"Time: {jax_jit_time:.4f} s")

# --- 4. PYTORCH MPS (GPU) ---
print("\n--- 3. PyTorch MPS (Apple GPU) ---")
if torch.backends.mps.is_available():
    dev_mps = torch.device("mps")
    x_mps = torch.randn(N, N, device=dev_mps)
    y_mps = torch.randn(N, N, device=dev_mps)

    # Warmup
    torch.mm(x_mps, y_mps)
    torch.mps.synchronize()

    start = time.time()
    for _ in range(LOOPS):
        torch.mm(x_mps, y_mps)
        torch.mps.synchronize()  # Critical for fair timing
    torch_mps_time = (time.time() - start) / LOOPS
    print(f"Time: {torch_mps_time:.4f} s")
else:
    torch_mps_time = None
    print("MPS not available.")

# --- 5. PYTORCH CPU ---
print("\n--- 4. PyTorch CPU ---")
dev_cpu = torch.device("cpu")
x_cpu = torch.randn(N, N, device=dev_cpu)
y_cpu = torch.randn(N, N, device=dev_cpu)

# Warmup
torch.mm(x_cpu, y_cpu)

start = time.time()
for _ in range(LOOPS):
    torch.mm(x_cpu, y_cpu)
    # CPU is synchronous by default, no special sync needed
torch_cpu_time = (time.time() - start) / LOOPS
print(f"Time: {torch_cpu_time:.4f} s")

# --- SUMMARY TABLE ---
print("\n" + "=" * 30)
print(f"{'Method':<20} | {'Time (s)':<10} | {'Rel Speed'}")
print("-" * 45)

# Use JAX JIT as the baseline (1.0x)
baseline = jax_jit_time


def fmt_speed(t):
    if t is None:
        return "N/A"
    return f"{baseline / t:.2f}x"


print(f"{'JAX JIT':<20} | {jax_jit_time:.4f}     | 1.00x (Baseline)")
print(f"{'JAX Eager':<20} | {jax_eager_time:.4f}     | {fmt_speed(jax_eager_time)}")
if torch_mps_time:
    print(
        f"{'PyTorch MPS':<20} | {torch_mps_time:.4f}     | {fmt_speed(torch_mps_time)}"
    )
print(f"{'PyTorch CPU':<20} | {torch_cpu_time:.4f}     | {fmt_speed(torch_cpu_time)}")
print("=" * 30)
