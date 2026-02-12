/**
 * Simple training example for microgpt-cpp
 * Based on Andrej Karpathy's microGPT: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include <microgpt/microgpt.h>
#include <iostream>
#include <iomanip>

using namespace microgpt;

int main() {
    // 1. Load dataset
    std::cout << "Loading dataset..." << std::endl;
    auto docs = load_docs("data/names.txt");
    if (docs.empty()) {
        std::cerr << "Error: Could not load data/names.txt" << std::endl;
        return 1;
    }
    shuffle(docs);
    std::cout << "num docs: " << docs.size() << std::endl;

    // 2. Create tokenizer
    Tokenizer tokenizer;
    tokenizer.fit(docs);
    std::cout << "vocab size: " << tokenizer.vocab_size << std::endl;

    // 3. Configure and initialize model
    Config config{
        .vocab_size = tokenizer.vocab_size,
        .n_embd = 16,
        .n_head = 4,
        .n_layer = 1,
        .block_size = 8
    };
    
    GPT model(config);
    std::cout << "num params: " << model.state_dict.get_all_params().size() << std::endl;

    // 4. Initialize optimizer
    Adam optimizer(1e-2, 0.9, 0.95, 1e-8);
    auto params = model.state_dict.get_all_params();
    optimizer.init(params.size());

    // 5. Train
    const int num_steps = 500;
    std::cout << "\nTraining..." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        // Create fresh storage for each step
        ValueStorage storage;
        
        // Sample a document and encode
        const std::string& doc = docs[step % docs.size()];
        const auto tokens = tokenizer.encode(doc);
        
        // Training step
        double loss = model.train_step(tokens, optimizer, storage);
        
        // Print progress
        if ((step + 1) % 10 == 0 || step == 0) {
            std::cout << "step " << std::setw(4) << (step + 1) << " / " << std::setw(4) << num_steps
                      << " | loss " << std::fixed << std::setprecision(4) << loss << std::endl;
        }
    }

    // 6. Save model
    std::cout << "\nSaving model..." << std::endl;
    model.save_weights("model_weights.bin", tokenizer);
    std::cout << "Model saved to model_weights.bin" << std::endl;
    std::cout << "Training complete!" << std::endl;

    return 0;
}
