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

    // Load tokenizer
    Tokenizer tokenizer;
    int uchars_size;
    infile.read(reinterpret_cast<char*>(&uchars_size), sizeof(int));
    tokenizer.uchars.resize(uchars_size);
    infile.read(reinterpret_cast<char*>(tokenizer.uchars.data()), uchars_size);
    infile.read(reinterpret_cast<char*>(&tokenizer.BOS), sizeof(int));
    tokenizer.vocab_size = config.vocab_size;

    // Initialize model
    GPT model(config);
    auto params = model.state_dict.get_all_params();

    // Load parameters
    for (auto* p : params) {
        infile.read(reinterpret_cast<char*>(&p->data), sizeof(double));
    }
    infile.close();

    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "vocab size: " << config.vocab_size << std::endl;
    std::cout << "num params: " << params.size() << std::endl;

    // Generate samples
    double temperature = 0.5;
    std::cout << "\n--- inference ---" << std::endl;

    for (int sample_idx = 0; sample_idx < 20; ++sample_idx) {
        auto tokens = model.generate(tokenizer.BOS, config.block_size, temperature);
        std::string sample = tokenizer.decode(tokens);
        std::cout << "sample " << std::setw(2) << (sample_idx + 1) << ": " << sample << std::endl;
    }

    return 0;
}
