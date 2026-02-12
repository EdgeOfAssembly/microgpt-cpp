# microgpt-cpp — Implementation Roadmap

This document tracks the phased port of [Andrej Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) from pure Python to optimized C++.

Work is split into **3 sequential PRs**, each building on the previous.

---

## PR 1 — Direct Python-to-C++ Translation
**Goal:** Faithful, working C++ port that mirrors the Python logic line-by-line.

- [x] Add original `microgpt.py` as reference
- [x] Basic `Tensor` class (contiguous float storage, shape, basic ops)
- [x] Scalar autograd `Value` class (direct port of Python's `Value`)
- [x] Port all layers: `RMSNorm`, `Linear`, `MultiHeadAttention`, `MLP`
- [x] Port `GPT` model class with `forward()` and `generate()`
- [x] Port `Adam` optimizer with bias correction
- [x] Port training loop (names.txt dataset)
- [x] Port inference / sampling
- [x] `examples/train.cpp` — train on names.txt, print loss
- [x] `examples/infer.cpp` — load weights, generate samples
- [x] `CMakeLists.txt` — basic build (no CUDA, no SIMD, no BLAS)
- [x] Verify output roughly matches Python on names dataset
- [x] Basic `README.md` with build and usage instructions

**Status:** ✅ Complete

---

## PR 2 — Header-Only / Single-Header API Refactor
**Goal:** Clean up into a reusable, header-only (or near header-only) C++ library with a public API.

- [x] Refactor into header-only structure under `include/microgpt/`:
  - [x] `layers.h` — RMSNorm, Linear layer functions
  - [x] `model.h` — GPT class with Config struct
  - [x] `optimizer.h` — Adam
  - [x] `utils.h` — softmax, sampling, tokenizer helpers
  - [x] `value.h` — Scalar autograd (kept, working well with graph-based approach)
  - [x] `microgpt.h` — Umbrella header that includes everything
- [x] Define clean public API:
  - [x] `GPT::GPT(Config)`
  - [x] `GPT::forward(token_id, pos) → logits`
  - [x] `GPT::generate(start_token, max_len, temperature)`
  - [x] `GPT::train_step(tokens, optimizer, storage)` — Single training step
  - [x] `GPT::save_weights(filename, tokenizer)` — Save model to binary
  - [x] `GPT::load_weights(filename)` — Load model from binary
- [x] Create educational examples with simple main():
  - [x] `train_simple.cpp` — 67-line training example
  - [x] `infer_simple.cpp` — 28-line inference example
- [x] Keep original detailed examples (`train.cpp`, `infer.cpp`) for reference
- [x] Update `CMakeLists.txt` for header-only install target
- [x] Ensure all examples still build and produce correct output

**Status:** ✅ Complete

---

## PR 3 — CUDA Acceleration + Optimized CPU Backend
**Goal:** Add CUDA backend with automatic fallback to an optimized CPU backend.

- [ ] Implement `backend.h` interface:
  - `CPUBackend` — matmul, softmax, RMSNorm, attention, activations
  - `CUDABackend` — same interface, GPU kernels
- [ ] CPU optimizations:
  - AVX2/SSE SIMD intrinsics for matmul and elementwise ops
  - OpenMP multi-threading for batch parallelism
  - Optional OpenBLAS / Accelerate linkage for matmul
- [ ] CUDA kernels:
  - Tiled matmul (shared memory)
  - Fused softmax
  - RMSNorm kernel
  - Scaled dot-product attention with causal mask + KV cache
  - ReLU² activation kernel
  - Optional cuBLAS fallback for matmul
- [ ] All CUDA code guarded with `#ifdef USE_CUDA`
- [ ] Runtime detection: auto-select GPU if available, fallback to CPU
- [ ] `src/backend_cpu.cpp` — CPU backend implementation
- [ ] `src/backend_cuda.cu` — CUDA backend implementation
- [ ] Update `CMakeLists.txt`:
  - `-DUSE_CUDA=ON` flag
  - `-DUSE_BLAS=ON` flag
  - Auto-detect CUDA toolkit
- [ ] Benchmark: CPU vs CUDA vs Python baseline
- [ ] Update `README.md` with:
  - CUDA build instructions
  - Benchmark results table
  - Architecture diagram
- [ ] Add MIT `LICENSE` file
- [ ] Final polish and cleanup

**Status:** 🔲 Not started

---

## Summary

| PR | Title | Depends On | Key Deliverable |
|----|-------|-----------|-----------------|
| 1  | Direct Python→C++ Translation | — | Working C++ port, matches Python output |
| 2  | Header-Only API Refactor | PR 1 | Clean reusable library, public API |
| 3  | CUDA + Optimized CPU Backend | PR 2 | GPU acceleration, SIMD CPU, benchmarks |

---

## Constraints (all PRs)
- Modern C++20, no external ML frameworks
- No PyTorch, no TensorRT
- Minimal dependencies (only optional: CUDA toolkit, OpenBLAS)
- Keep the spirit of microGPT: minimal, educational, readable
- Prioritize correctness first, then speed