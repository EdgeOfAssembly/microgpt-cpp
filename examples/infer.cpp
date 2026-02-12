/**
 * Detailed inference example for microgpt-cpp
 * Based on Andrej Karpathy's microGPT: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include <microgpt/microgpt.h>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace microgpt;

int main() {
    // Load model weights
    std::cout << "Loading model weights..." << std::endl;
    std::ifstream infile("model_weights.bin", std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not load model_weights.bin" << std::endl;
        std::cerr << "Please run ./train first to train the model." << std::endl;
        return 1;
    }

    // Load config
    Config config;
    infile.read(reinterpret_cast<char*>(&config.vocab_size), sizeof(int));
    infile.read(reinterpret_cast<char*>(&config.n_embd), sizeof(int));
    infile.read(reinterpret_cast<char*>(&config.n_head), sizeof(int));
    infile.read(reinterpret_cast<char*>(&config.n_layer), sizeof(int));
    infile.read(reinterpret_cast<char*>(&config.block_size), sizeof(int));
    
    // Validate loaded config
    assert(config.vocab_size > 0 && config.vocab_size < 10000 && "Invalid vocab size");
    assert(config.n_embd > 0 && config.n_embd < 10000 && "Invalid embedding dimension");
    assert(config.n_head > 0 && config.n_head < 1000 && "Invalid number of heads");
    assert(config.n_layer > 0 && config.n_layer < 1000 && "Invalid number of layers");
    assert(config.block_size > 0 && config.block_size < 10000 && "Invalid block size");
    assert(config.n_embd % config.n_head == 0 && "n_embd must be divisible by n_head");

    // Load tokenizer
    Tokenizer tokenizer;
    int uchars_size = 0;
    infile.read(reinterpret_cast<char*>(&uchars_size), sizeof(int));
    
    // Validate tokenizer size
    assert(uchars_size > 0 && uchars_size < 10000 && "Invalid tokenizer size");
    
    tokenizer.uchars.resize(uchars_size);
    infile.read(reinterpret_cast<char*>(tokenizer.uchars.data()), uchars_size);
    infile.read(reinterpret_cast<char*>(&tokenizer.BOS), sizeof(int));
    tokenizer.vocab_size = config.vocab_size;
    
    // Validate BOS token
    assert(tokenizer.BOS >= 0 && tokenizer.BOS < config.vocab_size && "Invalid BOS token");

    // Initialize model
    GPT model(config);
    auto params = model.state_dict.get_all_params();
    
    // Validate parameters
    for (const auto* param : params) {
        assert(param != nullptr && "Null parameter pointer");
    }

    // Load parameters
    for (auto* p : params) {
        assert(p != nullptr && "Null parameter pointer during load");
        infile.read(reinterpret_cast<char*>(&p->data), sizeof(double));
        
        // Validate loaded parameter
        if (!std::isfinite(p->data)) {
            std::cerr << "Warning: Loaded parameter with NaN or infinity" << std::endl;
        }
    }
    
    if (!infile) {
        std::cerr << "Error: Failed to read all parameters from file" << std::endl;
        return 1;
    }
    
    infile.close();

    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "vocab size: " << config.vocab_size << std::endl;
    std::cout << "num params: " << params.size() << std::endl;

    // Generate samples
    const double temperature = 0.5;
    std::cout << "\n--- inference ---" << std::endl;

    for (int sample_idx = 0; sample_idx < 20; ++sample_idx) {
        try {
            auto tokens = model.generate(tokenizer.BOS, config.block_size, temperature);
            std::string sample = tokenizer.decode(tokens);
            std::cout << "sample " << std::setw(2) << (sample_idx + 1) << ": " << sample << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error generating sample " << (sample_idx + 1) << ": " << e.what() << std::endl;
        }
    }

    return 0;
}
