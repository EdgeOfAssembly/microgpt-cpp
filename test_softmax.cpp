#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    ValueStorage storage;
    
    // Create logits
    Value* l1 = storage.store(Value(1.0));
    Value* l2 = storage.store(Value(2.0));
    Value* l3 = storage.store(Value(3.0));
    
    std::vector<Value*> logits = {l1, l2, l3};
    
    auto probs = softmax(logits, storage);
    
    std::cout << "Softmax results:" << std::endl;
    double sum = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        std::cout << "  probs[" << i << "] = " << probs[i]->data << std::endl;
        sum += probs[i]->data;
    }
    std::cout << "Sum = " << sum << " (should be 1.0)" << std::endl;
    
    // Test backward through softmax
    Value* loss = storage.store(-(probs[2]->log()));
    std::cout << "\nLoss (target=2): " << loss->data << std::endl;
    
    loss->backward();
    
    std::cout << "Gradients:" << std::endl;
    std::cout << "  l1->grad = " << l1->grad << std::endl;
    std::cout << "  l2->grad = " << l2->grad << std::endl;
    std::cout << "  l3->grad = " << l3->grad << std::endl;
    
    return 0;
}
