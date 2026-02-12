#include <microgpt/microgpt.h>
#include <iostream>

using namespace microgpt;

int main() {
    // Simple gradient test
    Value a(2.0);
    Value b(3.0);
    Value c = a * b + b.pow(2);  // c = 2*3 + 3^2 = 6 + 9 = 15
    
    std::cout << "c.data = " << c.data << " (expected 15)" << std::endl;
    
    c.backward();
    
    std::cout << "a.grad = " << a.grad << " (expected 3, dc/da = b = 3)" << std::endl;
    std::cout << "b.grad = " << b.grad << " (expected 8, dc/db = a + 2*b = 2 + 6 = 8)" << std::endl;
    
    return 0;
}
