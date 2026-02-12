#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    ValueStorage storage;
    
    // Simpler test - just 2 values
    Value* l1 = storage.store(Value(1.0));
    Value* l2 = storage.store(Value(2.0));
    
    std::vector<Value*> logits = {l1, l2};
    
    std::cout << "Calling softmax..." << std::endl;
    auto probs = softmax(logits, storage);
    
    std::cout << "Softmax done. probs[0] = " << probs[0]->data << std::endl;
    std::cout << "Softmax done. probs[1] = " << probs[1]->data << std::endl;
    
    std::cout << "Calling log..." << std::endl;
    Value* log_prob = storage.store(probs[1]->log());
    std::cout << "Log done: " << log_prob->data << std::endl;
    
    std::cout << "Negating..." << std::endl;
    Value* loss = storage.store(-*log_prob);
    std::cout << "Loss: " << loss->data << std::endl;
    
    std::cout << "Calling backward..." << std::endl;
    loss->backward();
    std::cout << "Backward done!" << std::endl;
    
    std::cout << "l1->grad = " << l1->grad << std::endl;
    std::cout << "l2->grad = " << l2->grad << std::endl;
    
    return 0;
}
