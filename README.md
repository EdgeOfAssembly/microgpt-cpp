# microgpt-cpp

A faithful C++20 port of [Andrej Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — the most minimal, dependency-free implementation of GPT training and inference.

This is a **line-by-line translation** from pure Python to modern C++20. No optimizations, no CUDA, no SIMD. The goal is correctness and matching Python behavior.

## Features

- **Scalar Autograd**: Direct port of the Python `Value` class with automatic differentiation
- **GPT Architecture**: Token + positional embeddings, multi-head attention, MLP with ReLU² activation, RMSNorm
- **Adam Optimizer**: With bias correction and cosine learning rate schedule
- **Character-level Tokenizer**: Simple encode/decode by character
- **Training & Inference**: Train on small text datasets and generate samples

## Requirements

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20 or higher
- No external dependencies

## Build Instructions

```bash
# Clone the repository
git clone https://github.com/EdgeOfAssembly/microgpt-cpp.git
cd microgpt-cpp

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Or on Windows with MSVC:
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

## Usage

### Training

Train a character-level GPT on the names dataset:

```bash
# From the build directory
./train
```

This will:
1. Load `data/names.txt` (a dataset of ~32,000 names)
2. Train a small GPT model (16 embedding dimensions, 4 attention heads, 1 layer)
3. Print loss every 10 steps for 500 steps
4. Save trained weights to `model_weights.bin`

Expected output:
```
Loading dataset...
num docs: 32033
vocab size: 27
Initializing model...
num params: 1324
Training...
step    1 /  500 | loss 3.2943
step   10 /  500 | loss 2.8571
...
step  500 /  500 | loss 1.9234
Model saved to model_weights.bin
Training complete!
```

### Inference

Generate sample names from the trained model:

```bash
# From the build directory
./infer
```

This will:
1. Load the trained weights from `model_weights.bin`
2. Generate 20 sample names using temperature 0.5

Expected output:
```
Loading model weights...
Model loaded successfully!
vocab size: 27
num params: 1324

--- inference ---
sample  1: marley
sample  2: kaden
sample  3: ashlyn
...
sample 20: jayden
```

## Project Structure

```
microgpt-cpp/
├── microgpt.py              # Original Python reference
├── data/
│   └── names.txt            # Training dataset
├── include/microgpt/
│   ├── microgpt.h           # Main header (includes all)
│   ├── value.h              # Scalar autograd Value class
│   ├── utils.h              # Utilities (tokenizer, softmax, etc.)
│   ├── model.h              # GPT model class
│   └── optimizer.h          # Adam optimizer
├── examples/
│   ├── train.cpp            # Training example
│   └── infer.cpp            # Inference example
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
└── TODO.md                  # Implementation roadmap

```

## Model Architecture

The GPT model follows the architecture from the original microGPT:

- **Embeddings**: Token embeddings + positional embeddings
- **Transformer Layers** (configurable):
  - Pre-LayerNorm with RMSNorm
  - Multi-head causal self-attention
  - Residual connections
  - MLP with ReLU² activation
- **Output**: Linear projection to vocabulary logits

## Algorithm Overview

1. **Tokenization**: Character-level vocabulary (each unique character is a token)
2. **Forward Pass**: Input sequence → embeddings → transformer layers → logits
3. **Loss**: Cross-entropy on next-token prediction
4. **Backward Pass**: Scalar autograd computes gradients via chain rule
5. **Optimization**: Adam with cosine learning rate decay
6. **Generation**: Autoregressive sampling with temperature

## Differences from Python

This C++ implementation is a faithful port with these minimal changes:

- Uses `std::vector` instead of Python lists
- Uses `double` instead of Python floats
- Uses `std::map` for state_dict instead of Python dict
- Memory management is explicit (C++ Value objects manage their own children pointers)
- Random number generation uses `<random>` instead of Python's `random` module

## Performance

This is an **educational implementation** optimized for readability and correctness, not speed. Training 500 steps takes roughly:

- Python baseline: ~30-60 seconds
- C++ (this implementation): ~60-120 seconds

Performance improvements (CUDA, SIMD, batching) will come in future PRs.

## License

MIT License - See LICENSE file for details

## Credits

- Original microGPT Python implementation by [Andrej Karpathy](https://twitter.com/karpathy)
- Names dataset from [makemore](https://github.com/karpathy/makemore)

## References

- [Original microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)