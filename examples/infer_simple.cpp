/**
 * Simple inference example for microgpt-cpp
 * Based on Andrej Karpathy's microGPT: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include <microgpt/microgpt.h>
#include <iostream>
#include <iomanip>

using namespace microgpt;

int main() {
    // 1. Load model and tokenizer
    std::cout << "Loading model..." << std::endl;
    auto [model, tokenizer] = GPT::load_weights("model_weights.bin");
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "vocab size: " << model.config.vocab_size << std::endl;
    std::cout << "num params: " << model.state_dict.get_all_params().size() << std::endl;

    // 2. Generate samples
    const double temperature = 0.5;
    std::cout << "\n--- inference ---" << std::endl;

    for (int i = 0; i < 20; ++i) {
        auto tokens = model.generate(tokenizer.BOS, model.config.block_size, temperature);
        std::string sample = tokenizer.decode(tokens);
        std::cout << "sample " << std::setw(2) << (i + 1) << ": " << sample << std::endl;
    }

    return 0;
}
