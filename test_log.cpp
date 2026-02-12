#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    ValueStorage storage;
    
    Value* a = storage.store(Value(2.0));
    std::cout << "a = " << a->data << std::endl;
    
    Value* b = storage.store(a->log());
    std::cout << "b = log(a) = " << b->data << std::endl;
    
    Value* c = storage.store(-*b);
    std::cout << "c = -b = " << c->data << std::endl;
    
    std::cout << "Before backward" << std::endl;
    c->backward();
    std::cout << "After backward" << std::endl;
    
    std::cout << "a->grad = " << a->grad << std::endl;
    
    return 0;
}
