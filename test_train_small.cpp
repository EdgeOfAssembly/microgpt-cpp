#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    std::cout << "Testing training with factory methods..." << std::endl;
    
    auto docs = load_docs("data/names.txt");
    if (docs.empty()) {
        std::cerr << "Could not load data" << std::endl;
        return 1;
    }
    shuffle(docs);
    std::cout << "Loaded " << docs.size() << " docs" << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.fit(docs);
    std::cout << "Vocab size: " << tokenizer.vocab_size << std::endl;
    
    Config config;
    config.vocab_size = tokenizer.vocab_size;
    config.n_embd = 16;
    config.n_head = 4;
    config.n_layer = 1;
    config.block_size = 8;
    
    GPT model(config);
    auto params = model.state_dict.get_all_params();
    std::cout << "Num params: " << params.size() << std::endl;
    
    Adam optimizer(1e-2, 0.9, 0.95, 1e-8);
    optimizer.init(params.size());
    
    // Just 5 training steps
    const int num_steps = 5;
    std::cout << "\nTraining " << num_steps << " steps..." << std::endl;
    
    for (int step = 0; step < num_steps; ++step) {
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
            const int token_id = tokens[pos_id];
            const int target_id = tokens[pos_id + 1];
            
            auto logits = model.forward(token_id, pos_id, keys, values, storage);
            auto probs = softmax(logits, storage);
            
            Value* log_prob = storage.log(probs[target_id]);
            Value* loss_t = storage.neg(log_prob);
            losses.push_back(loss_t);
        }
        
        Value* loss = storage.constant(0.0);
        for (const auto* l : losses) {
            loss = storage.add(loss, const_cast<Value*>(l));
        }
        
        Value* n_val = storage.constant(static_cast<double>(n));
        loss = storage.div(loss, n_val);
        
        std::cout << "Step " << (step + 1) << ": forward pass complete, loss = " << loss->data << std::endl;
        
        loss->backward();
        std::cout << "         backward pass complete" << std::endl;
        
        optimizer.step(params, num_steps);
        std::cout << "         optimizer step complete" << std::endl;
    }
    
    std::cout << "\n✅ Training completed successfully!" << std::endl;
    return 0;
}
