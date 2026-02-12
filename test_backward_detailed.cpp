#include <microgpt/microgpt.h>
#include <iostream>
#include <csignal>

void signal_handler(int signal) {
    std::cerr << "\nCaught signal " << signal << " (segmentation fault)" << std::endl;
    std::exit(1);
}

using namespace microgpt;

int main() {
    signal(SIGSEGV, signal_handler);
    
    ValueStorage storage;
    
    // Simple test
    Value* a = storage.store(Value(2.0));
    Value* b = storage.store(Value(3.0));
    Value* c = storage.store(*a + *b);
    
    std::cout << "Simple test: c = " << c->data << std::endl;
    std::cout << "Calling backward on simple graph..." << std::endl;
    c->backward();
    std::cout << "Simple backward OK. a->grad = " << a->grad << std::endl;
    
    // Now test with more complex operations
    ValueStorage storage2;
    Value* x = storage2.store(Value(1.0));
    Value* y = storage2.store(Value(2.0));
    
    std::cout << "\nTesting pow..." << std::endl;
    Value* y_inv = storage2.store(y->pow(-1));
    std::cout << "y_inv = " << y_inv->data << std::endl;
    
    std::cout << "Testing multiply with pow result..." << std::endl;
    Value* z = storage2.store(*x * *y_inv);
    std::cout << "z = x * y_inv = " << z->data << std::endl;
    
    std::cout << "Calling backward..." << std::endl;
    z->backward();
    std::cout << "Backward OK!" << std::endl;
    std::cout << "x->grad = " << x->grad << std::endl;
    std::cout << "y->grad = " << y->grad << std::endl;
    
    return 0;
}
