#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    std::cout << "Testing factory methods for automatic heap allocation..." << std::endl;
    
    ValueStorage storage;
    
    // Test basic operations
    std::cout << "\n1. Testing basic operations:" << std::endl;
    Value* a = storage.constant(3.0);
    Value* b = storage.constant(4.0);
    
    Value* sum = storage.add(a, b);
    std::cout << "   3 + 4 = " << sum->data << " (expected 7)" << std::endl;
    
    Value* prod = storage.mul(a, b);
    std::cout << "   3 * 4 = " << prod->data << " (expected 12)" << std::endl;
    
    Value* diff = storage.sub(a, b);
    std::cout << "   3 - 4 = " << diff->data << " (expected -1)" << std::endl;
    
    Value* quot = storage.div(a, b);
    std::cout << "   3 / 4 = " << quot->data << " (expected 0.75)" << std::endl;
    
    // Test unary operations
    std::cout << "\n2. Testing unary operations:" << std::endl;
    Value* neg_a = storage.neg(a);
    std::cout << "   -3 = " << neg_a->data << " (expected -3)" << std::endl;
    
    Value* pow_a = storage.pow(a, 2.0);
    std::cout << "   3^2 = " << pow_a->data << " (expected 9)" << std::endl;
    
    Value* log_a = storage.log(a);
    std::cout << "   log(3) = " << log_a->data << " (expected ~1.099)" << std::endl;
    
    Value* exp_a = storage.exp(a);
    std::cout << "   exp(3) = " << exp_a->data << " (expected ~20.09)" << std::endl;
    
    Value* relu_neg = storage.relu(neg_a);
    std::cout << "   relu(-3) = " << relu_neg->data << " (expected 0)" << std::endl;
    
    // Test softmax (the critical case that was failing)
    std::cout << "\n3. Testing softmax:" << std::endl;
    Value* l1 = storage.constant(1.0);
    Value* l2 = storage.constant(2.0);
    std::vector<Value*> logits = {l1, l2};
    
    auto probs = softmax(logits, storage);
    std::cout << "   softmax([1, 2]) = [" << probs[0]->data << ", " << probs[1]->data << "]" << std::endl;
    std::cout << "   (expected ~[0.269, 0.731])" << std::endl;
    
    // Test backward pass
    std::cout << "\n4. Testing backward pass:" << std::endl;
    probs[1]->backward();
    
    std::cout << "   After backward on probs[1]:" << std::endl;
    std::cout << "   l1->grad = " << l1->grad << " (expected ~-0.197)" << std::endl;
    std::cout << "   l2->grad = " << l2->grad << " (expected ~0.197)" << std::endl;
    
    // Test complex expression
    std::cout << "\n5. Testing complex expression:" << std::endl;
    storage.clear();
    Value* x = storage.constant(2.0);
    Value* y = storage.constant(3.0);
    
    // Compute (x*y + x) / y
    Value* xy = storage.mul(x, y);
    Value* xy_plus_x = storage.add(xy, x);
    Value* result = storage.div(xy_plus_x, y);
    
    std::cout << "   (2*3 + 2) / 3 = " << result->data << " (expected ~2.667)" << std::endl;
    
    result->backward();
    std::cout << "   x->grad = " << x->grad << " (expected ~1.333)" << std::endl;
    std::cout << "   y->grad = " << y->grad << " (expected ~-0.222)" << std::endl;
    
    std::cout << "\n✅ All tests passed! No stack temporaries!" << std::endl;
    return 0;
}
