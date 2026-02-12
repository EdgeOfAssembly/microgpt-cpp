#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    // Test the pointer-based API
    ValueStorage storage;
    
    // Create simple values
    Value* a = storage.store(Value(2.0));
    Value* b = storage.store(Value(3.0));
    
    // Test addition
    Value* c = storage.store(*a + *b);
    std::cout << "c = a + b = " << c->data << " (expected 5)" << std::endl;
    
    // Test backward
    c->backward();
    std::cout << "a->grad = " << a->grad << " (expected 1)" << std::endl;
    std::cout << "b->grad = " << b->grad << " (expected 1)" << std::endl;
    
    // Test linear function
    std::vector<Value*> x = {a, b};
    std::vector<std::vector<Value>> w = {{Value(1.0), Value(2.0)}, {Value(3.0), Value(4.0)}};
    
    ValueStorage storage2;
    auto result = linear(x, w, storage2);
    
    std::cout << "linear result[0] = " << result[0]->data << " (expected 1*2 + 2*3 = 8)" << std::endl;
    std::cout << "linear result[1] = " << result[1]->data << " (expected 3*2 + 4*3 = 18)" << std::endl;
    
    return 0;
}
