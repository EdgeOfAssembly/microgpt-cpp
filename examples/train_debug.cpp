#include <microgpt/microgpt.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

using namespace microgpt;

void signal_handler(int sig) {
    std::cerr << "\n=== CAUGHT SIGNAL " << sig << " ===\n";
    void* array[50];
    size_t size = backtrace(array, 50);
    std::cerr << "Backtrace:\n";
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    std::exit(1);
}

int main() {
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);
    
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

    // Model configuration
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

    // Training loop - just 2 steps for debugging
    const int num_steps = 2;
    std::cout << "\nTraining..." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        std::cout << "=== Step " << step << " ===" << std::endl;
        ValueStorage storage;
        
        const std::string& doc = docs[step % docs.size()];
        const auto tokens = tokenizer.encode(doc);
        const int n = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);

        if (n <= 0) continue;

        std::vector<std::vector<std::vector<Value*>>> keys(config.n_layer);
        std::vector<std::vector<std::vector<Value*>>> values(config.n_layer);
        std::vector<Value*> losses;
        losses.reserve(n);

        for (int pos_id = 0; pos_id < n; ++pos_id) {
            std::cout << "  pos " << pos_id << std::endl;
            const int token_id = tokens[pos_id];
            const int target_id = tokens[pos_id + 1];

            std::cout << "    forward..." << std::endl;
            auto logits = model.forward(token_id, pos_id, keys, values, storage);
            
            std::cout << "    softmax..." << std::endl;
            auto probs = softmax(logits, storage);
            
            std::cout << "    loss..." << std::endl;
            Value* log_prob = storage.log(probs[target_id]);
            Value* neg_log = storage.neg(log_prob);
            losses.push_back(neg_log);
        }

        std::cout << "  averaging loss..." << std::endl;
        Value* loss = storage.constant(0.0);
        for (const auto* l : losses) {
            loss = storage.add(loss, l);
        }
        Value* n_val = storage.constant(static_cast<double>(n));
        Value* n_inv = storage.pow(n_val, -1);
        loss = storage.mul(loss, n_inv);

        std::cout << "  backward..." << std::endl;
        loss->backward();

        std::cout << "  optimizer step..." << std::endl;
        optimizer.step(params, num_steps);

        std::cout << "step " << (step + 1) << " | loss " << std::fixed << std::setprecision(4) << loss->data << std::endl;
    }

    std::cout << "\nTraining complete!" << std::endl;
    return 0;
}
