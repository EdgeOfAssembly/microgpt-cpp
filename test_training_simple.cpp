#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    // Very simple test: train on a single token sequence
    // Create a tiny vocab and model
    Config config;
    config.vocab_size = 5;  // Just 5 tokens
    config.n_embd = 4;
    config.n_head = 1;
    config.n_layer = 1;
    config.block_size = 3;
    
    GPT model(config);
    auto params = model.state_dict.get_all_params();
    std::cout << "Num params: " << params.size() << std::endl;
    
    // Check initial gradients
    std::cout << "Initial gradients (should be 0):" << std::endl;
    for (int i = 0; i < std::min(5, (int)params.size()); ++i) {
        std::cout << "  param[" << i << "].grad = " << params[i]->grad << std::endl;
    }
    
    // Simple forward pass
    ValueStorage storage;
    std::vector<std::vector<std::vector<Value*>>> keys(1);
    std::vector<std::vector<std::vector<Value*>>> values(1);
    
    // Forward pass with tokens [0, 1]
    auto logits0 = model.forward(0, 0, keys, values, storage);
    auto probs0 = softmax(logits0, storage);
    Value* log_prob = storage.log(probs0[1]);
    Value* loss0 = storage.neg(log_prob);  // Target is token 1
    
    std::cout << "\nForward pass done. Loss = " << loss0->data << std::endl;
    
    // Backward
    loss0->backward();
    
    std::cout << "\nBackward pass done. Checking gradients:" << std::endl;
    int non_zero = 0;
    for (int i = 0; i < std::min(10, (int)params.size()); ++i) {
        std::cout << "  param[" << i << "].grad = " << params[i]->grad << std::endl;
        if (params[i]->grad != 0.0) non_zero++;
    }
    std::cout << "Total non-zero gradients in first 10: " << non_zero << std::endl;
    
    return 0;
}
