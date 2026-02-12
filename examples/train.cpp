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

    // Initialize model
    std::cout << "Initializing model..." << std::endl;
    GPT model(config);
    auto params = model.state_dict.get_all_params();
    std::cout << "num params: " << params.size() << std::endl;

    // Initialize optimizer
    Adam optimizer(1e-2, 0.9, 0.95, 1e-8);
    optimizer.init(params.size());

    // Training loop
    int num_steps = 500;
    std::cout << "\nTraining..." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        // Sample a document
        const std::string& doc = docs[step % docs.size()];
        auto tokens = tokenizer.encode(doc);
        int n = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);

        // Forward pass
        std::vector<std::vector<std::vector<Value>>> keys(config.n_layer);
        std::vector<std::vector<std::vector<Value>>> values(config.n_layer);
        std::vector<Value> losses;

        for (int pos_id = 0; pos_id < n; ++pos_id) {
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];

            auto logits = model.forward(token_id, pos_id, keys, values);
            auto probs = softmax(logits);
            Value loss_t = -(probs[target_id].log());
            losses.push_back(loss_t);
        }

        // Average loss
        Value loss(0.0);
        for (const auto& l : losses) {
            loss = loss + l;
        }
        loss = loss / static_cast<double>(n);

        // Backward pass
        loss.backward();

        // Optimizer step
        optimizer.step(params, num_steps);

        // Print progress
        if ((step + 1) % 10 == 0 || step == 0) {
            std::cout << "step " << std::setw(4) << (step + 1) << " / " << std::setw(4) << num_steps
                      << " | loss " << std::fixed << std::setprecision(4) << loss.data << std::endl;
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
