#include <microgpt/microgpt.h>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace microgpt;

int main() {
    // Load dataset
    std::cout << "Loading dataset..." << std::endl;
    auto docs = load_docs("data/names.txt");
    if (docs.empty()) {
        std::cerr << "Error: Could not load data/names.txt" << std::endl;
        return 1;
    }
    shuffle(docs);
    std::cout << "num docs: " << docs.size() << std::endl;

    // Create tokenizer
    Tokenizer tokenizer;
    tokenizer.fit(docs);
    std::cout << "vocab size: " << tokenizer.vocab_size << std::endl;

    // Model configuration (matching Python version)
    Config config;
    config.vocab_size = tokenizer.vocab_size;
    config.n_embd = 16;
    config.n_head = 4;
    config.n_layer = 1;
    config.block_size = 8;

    // Validate configuration
    assert(config.vocab_size > 0 && "Invalid vocab size");
    assert(config.n_embd > 0 && "Invalid embedding dimension");
    assert(config.n_head > 0 && "Invalid number of heads");
    assert(config.n_layer > 0 && "Invalid number of layers");
    assert(config.block_size > 0 && "Invalid block size");
    assert(config.n_embd % config.n_head == 0 && "n_embd must be divisible by n_head");

    // Initialize model
    std::cout << "Initializing model..." << std::endl;
    GPT model(config);
    auto params = model.state_dict.get_all_params();
    
    // Validate all parameters
    for (const auto* param : params) {
        assert(param != nullptr && "Null parameter pointer detected");
    }
    
    std::cout << "num params: " << params.size() << std::endl;

    // Initialize optimizer
    Adam optimizer(1e-2, 0.9, 0.95, 1e-8);
    optimizer.init(params.size());

    // Training loop
    const int num_steps = 500;
    std::cout << "\nTraining..." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        // Create storage for this training step
        ValueStorage storage;
        
        // Sample a document
        const std::string& doc = docs[step % docs.size()];
        const auto tokens = tokenizer.encode(doc);
        const int n = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);

        if (n <= 0) {
            continue;  // Skip empty sequences
        }

        // Forward pass
        std::vector<std::vector<std::vector<Value*>>> keys(config.n_layer);
        std::vector<std::vector<std::vector<Value*>>> values(config.n_layer);
        std::vector<Value*> losses;
        losses.reserve(n);

        for (int pos_id = 0; pos_id < n; ++pos_id) {
            const int token_id = tokens[pos_id];
            const int target_id = tokens[pos_id + 1];

            // Bounds checking
            assert(token_id >= 0 && token_id < config.vocab_size && "Token ID out of range");
            assert(target_id >= 0 && target_id < config.vocab_size && "Target ID out of range");

            auto logits = model.forward(token_id, pos_id, keys, values, storage);
            
            // Validate logits
            assert(!logits.empty() && "Forward pass returned empty logits");
            for (const auto* logit : logits) {
                assert(logit != nullptr && "Null logit pointer");
            }
            
            auto probs = softmax(logits, storage);
            
            // Validate probs
            assert(static_cast<int>(probs.size()) == config.vocab_size && "Probability size mismatch");
            assert(probs[target_id] != nullptr && "Null probability pointer at target");
            
            Value* loss_t = storage.store(-(probs[target_id]->log()));
            assert(loss_t != nullptr && "Null loss pointer");
            losses.push_back(loss_t);
        }

        // Average loss
        Value* loss = storage.store(Value(0.0));
        assert(loss != nullptr && "Null loss accumulator");
        
        for (const auto* l : losses) {
            assert(l != nullptr && "Null loss in losses vector");
            loss = storage.store(*loss + *l);
            assert(loss != nullptr && "Null loss after accumulation");
        }
        
        Value* n_val = storage.store(Value(static_cast<double>(n)));
        assert(n_val != nullptr && "Null n_val");
        loss = storage.store(*loss / *n_val);
        assert(loss != nullptr && "Null loss after averaging");

        // Backward pass
        try {
            loss->backward();
        } catch (const std::exception& e) {
            std::cerr << "Error in backward pass at step " << step << ": " << e.what() << std::endl;
            return 1;
        }

        // Optimizer step
        optimizer.step(params, num_steps);

        // Print progress
        if ((step + 1) % 10 == 0 || step == 0) {
            std::cout << "step " << std::setw(4) << (step + 1) << " / " << std::setw(4) << num_steps
                      << " | loss " << std::fixed << std::setprecision(4) << loss->data << std::endl;
        }
    }

    // Save model weights
    std::cout << "\nSaving model weights..." << std::endl;
    std::ofstream outfile("model_weights.bin", std::ios::binary);
    if (outfile.is_open()) {
        // Save config
        outfile.write(reinterpret_cast<const char*>(&config.vocab_size), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.n_embd), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.n_head), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.n_layer), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.block_size), sizeof(int));

        // Save tokenizer
        int uchars_size = static_cast<int>(tokenizer.uchars.size());
        outfile.write(reinterpret_cast<const char*>(&uchars_size), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(tokenizer.uchars.data()), uchars_size);
        outfile.write(reinterpret_cast<const char*>(&tokenizer.BOS), sizeof(int));

        // Save parameters
        for (const auto* p : params) {
            assert(p != nullptr && "Null parameter during save");
            outfile.write(reinterpret_cast<const char*>(&p->data), sizeof(double));
        }
        outfile.close();
        std::cout << "Model saved to model_weights.bin" << std::endl;
    } else {
        std::cerr << "Error: Could not save model weights" << std::endl;
    }

    std::cout << "\nTraining complete!" << std::endl;
    return 0;
}
