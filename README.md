# microgpt-cpp

A faithful C++20 port of [Andrej Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — the most minimal, dependency-free implementation of GPT training and inference.

This is a **line-by-line translation** from pure Python to modern C++20. No optimizations, no CUDA, no SIMD. The goal is correctness and matching Python behavior.

**Original work by:** [Andrej Karpathy](https://twitter.com/karpathy)  
**Original Python implementation:** https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

## Documentation

- **Man Page**: See `man microgpt-cpp` after installation, or view `docs/microgpt-cpp.7`
- **API Reference**: Comprehensive documentation in the man page
- **Examples**: Both simple and detailed examples in `examples/` directory

## Features

- **Scalar Autograd**: Direct port of the Python `Value` class with automatic differentiation
- **GPT Architecture**: Token + positional embeddings, multi-head attention, MLP with ReLU² activation, RMSNorm
- **Adam Optimizer**: With bias correction and cosine learning rate schedule
- **Character-level Tokenizer**: Simple encode/decode by character
- **Training & Inference**: Train on small text datasets and generate samples

## Prerequisites

- `binutils`
- `g++` (must support C++20)
- `git`
- `cmake` (3.20 or higher)
- `make` or optionally `ninja`

## Build Instructions

```bash
# Clone the repository
git clone git@github.com:EdgeOfAssembly/microgpt-cpp.git
cd microgpt-cpp

# Default build with GNU Make
cmake -B build && make -j$(nproc) -C build

# Or optionally with Ninja
cmake -B build -G Ninja && ninja -C build

# Run from build directory
cd build
./train
./infer
```

## Usage

### Simple Training Example

The simplified API makes training a model incredibly straightforward:

```cpp
#include <microgpt/microgpt.h>
using namespace microgpt;

int main() {
    // 1. Load dataset and create tokenizer
    auto docs = load_docs("data/names.txt");
    shuffle(docs);
    Tokenizer tokenizer;
    tokenizer.fit(docs);
    
    // 2. Configure and initialize model
    Config config{.vocab_size = tokenizer.vocab_size, .n_embd = 16, 
                  .n_head = 4, .n_layer = 1, .block_size = 8};
    GPT model(config);
    
    // 3. Initialize optimizer
    Adam optimizer(1e-2, 0.9, 0.95, 1e-8);
    optimizer.init(model.state_dict.get_all_params().size());
    
    // 4. Train
    for (int step = 0; step < 500; ++step) {
        ValueStorage storage;  // Fresh storage each step
        const auto tokens = tokenizer.encode(docs[step % docs.size()]);
        double loss = model.train_step(tokens, optimizer, storage);
        if ((step + 1) % 10 == 0) {
            std::cout << "step " << (step + 1) << " | loss " << loss << std::endl;
        }
    }
    
    // 5. Save model
    model.save_weights("model_weights.bin", tokenizer);
    return 0;
}
```

See [`examples/train_simple.cpp`](examples/train_simple.cpp) for the complete example.

### Simple Inference Example

Loading and using a trained model is equally simple:

```cpp
#include <microgpt/microgpt.h>
using namespace microgpt;

int main() {
    // 1. Load model and tokenizer
    auto [model, tokenizer] = GPT::load_weights("model_weights.bin");
    
    // 2. Generate samples
    for (int i = 0; i < 20; ++i) {
        auto tokens = model.generate(tokenizer.BOS, model.config.block_size, 0.5);
        std::string sample = tokenizer.decode(tokens);
        std::cout << "sample " << (i + 1) << ": " << sample << std::endl;
    }
    return 0;
}
```

See [`examples/infer_simple.cpp`](examples/infer_simple.cpp) for the complete example.

### Detailed Examples

For more control and understanding of the training loop internals:
- [`examples/train.cpp`](examples/train.cpp) — Full training loop with detailed loss computation
- [`examples/infer.cpp`](examples/infer.cpp) — Detailed inference with manual weight loading

## Training

Train a character-level GPT on the names dataset:

```bash
# From the build directory
./train_simple
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
num params: 4064

Training...
step    1 /  500 | loss 3.3207
step   10 /  500 | loss 3.2782
...
step  500 /  500 | loss 1.7959

Saving model...
Model saved to model_weights.bin
Training complete!
```

## Inference

Generate sample names from the trained model:

```bash
# From the build directory
./infer_simple
```

This will:
1. Load the trained weights from `model_weights.bin`
2. Generate 20 sample names using temperature 0.5

Expected output:
```
Loading model...
Model loaded successfully!
vocab size: 27
num params: 4064

--- inference ---
sample  1: licel
sample  2: amilia
sample  3: janar
...
sample 20: akaren
```

## Public API

### Core Classes

#### `Config`
Model configuration structure:
```cpp
struct Config {
    int vocab_size;   // Vocabulary size
    int n_embd;       // Embedding dimension
    int n_head;       // Number of attention heads
    int n_layer;      // Number of transformer layers
    int block_size;   // Maximum sequence length
};
```

#### `GPT`
Main model class with simple, educational API:

**Constructor:**
```cpp
GPT(const Config& config);  // Initialize model with given config
```

**Training:**
```cpp
double train_step(const std::vector<int>& tokens, Adam& optimizer, ValueStorage& storage);
// Performs one training step on a sequence
// Returns the loss value
```

**Inference:**
```cpp
std::vector<int> generate(int start_token, int max_length, double temperature = 1.0);
// Generate a sequence autoregressively
// Returns vector of generated token IDs
```

**Weight Persistence:**
```cpp
void save_weights(const std::string& filename, const Tokenizer& tokenizer) const;
// Save model weights and config to binary file

static std::pair<GPT, Tokenizer> load_weights(const std::string& filename);
// Load model and tokenizer from binary file
```

#### `Tokenizer`
Simple character-level tokenizer:
```cpp
void fit(const std::vector<std::string>& docs);           // Build vocabulary
std::vector<int> encode(const std::string& text) const;   // Text → tokens
std::string decode(const std::vector<int>& tokens) const; // Tokens → text
```

#### `Adam`
Adam optimizer with cosine learning rate schedule:
```cpp
Adam(double lr, double beta1, double beta2, double eps);
void init(size_t num_params);
void step(std::vector<Value*>& params, int total_steps);
```

### Utility Functions

```cpp
std::vector<std::string> load_docs(const std::string& filename);  // Load text file
void shuffle(std::vector<std::string>& docs);                     // Shuffle dataset
std::vector<Value*> softmax(const std::vector<Value*>& logits, ValueStorage& storage);
```

### Layer Functions

```cpp
std::vector<Value*> rmsnorm(const std::vector<Value*>& x, ValueStorage& storage);
std::vector<Value*> linear(const std::vector<Value*>& x, 
                            const std::vector<std::vector<Value>>& w, 
                            ValueStorage& storage);
```

## Project Structure

```
microgpt-cpp/
├── microgpt.py              # Original Python reference
├── data/
│   └── names.txt            # Training dataset
├── include/microgpt/
│   ├── microgpt.h           # Main header (includes all)
│   ├── value.h              # Scalar autograd Value class + ValueStorage
│   ├── layers.h             # Layer functions (RMSNorm, Linear)
│   ├── utils.h              # Utilities (tokenizer, softmax, etc.)
│   ├── model.h              # GPT model class with clean API
│   └── optimizer.h          # Adam optimizer
├── examples/
│   ├── train_simple.cpp     # Simple training example (67 lines)
│   ├── infer_simple.cpp     # Simple inference example (28 lines)
│   ├── train.cpp            # Detailed training example
│   └── infer.cpp            # Detailed inference example
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

This is an educational implementation using scalar autograd, optimized for readability and correctness — not speed. Performance is comparable to the Python baseline.

High-performance tensor operations with SIMD/CUDA acceleration are planned for PR 3.

## License

MIT License - See LICENSE file for details

## Credits

- Original microGPT Python implementation by [Andrej Karpathy](https://twitter.com/karpathy)
- Names dataset from [makemore](https://github.com/karpathy/makemore)

## References

- [Original microGPT gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)