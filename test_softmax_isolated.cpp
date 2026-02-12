#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    ValueStorage storage;
    
    // Create 2 logits
    Value* l1 = storage.store(Value(1.0));
    Value* l2 = storage.store(Value(2.0));
    
    std::vector<Value*> logits = {l1, l2};
    
    std::cout << "Calling softmax..." << std::endl;
    auto probs = softmax(logits, storage);
    
    std::cout << "Softmax done:" << std::endl;
    std::cout << "  probs[0] = " << probs[0]->data << std::endl;
    std::cout << "  probs[1] = " << probs[1]->data << std::endl;
    std::cout << "  Storage size: " << storage.size() << std::endl;
    
    // Try backward on probs[1] directly (without log)
    std::cout << "\nCalling backward on probs[1]..." << std::endl;
    try {
        probs[1]->backward();
        std::cout << "Backward successful!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    
    std::cout << "l1->grad = " << l1->grad << std::endl;
    std::cout << "l2->grad = " << l2->grad << std::endl;
    
    return 0;
}
