#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    ValueStorage storage;
    
    // Create simple computation: c = a + b
    Value* a = storage.store(Value(2.0));
    Value* b = storage.store(Value(3.0));
    Value* c = storage.store(*a + *b);
    
    std::cout << "c.data = " << c->data << " (expected 5)" << std::endl;
    
    std::cout << "Before backward:" << std::endl;
    std::cout << "  a->grad = " << a->grad << std::endl;
    std::cout << "  b->grad = " << b->grad << std::endl;
    std::cout << "  c->grad = " << c->grad << std::endl;
    
    try {
        c->backward();
        std::cout << "Backward completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    
    std::cout << "After backward:" << std::endl;
    std::cout << "  a->grad = " << a->grad << std::endl;
    std::cout << "  b->grad = " << b->grad << std::endl;
    std::cout << "  c->grad = " << c->grad << std::endl;
    
    return 0;
}
