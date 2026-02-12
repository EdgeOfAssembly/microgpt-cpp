#include <iostream>
#include <vector>
#include <cmath>

// Simplified Value class for testing
class Value {
public:
    double data;
    double grad;
    std::vector<Value*> children;
    std::vector<double> local_grads;
    
    Value(double d) : data(d), grad(0) {}
    Value(double d, std::vector<Value*> c, std::vector<double> lg) 
        : data(d), grad(0), children(c), local_grads(lg) {}
    
    void backward() {
        grad = 1.0;
        // Simple backward - just accumulate grads
        for (size_t i = 0; i < children.size(); ++i) {
            std::cout << "Accumulating grad " << (local_grads[i] * grad) 
                      << " to child " << i << " at " << children[i] << std::endl;
            children[i]->grad += local_grads[i] * grad;
        }
    }
};

int main() {
    Value a(2.0);
    Value b(3.0);
    
    std::cout << "a at " << &a << ", b at " << &b << std::endl;
    
    // Create a*b 
    Value temp1(a.data * b.data, {&a, &b}, {b.data, a.data});
    std::cout << "temp1 = a*b = " << temp1.data << " at " << &temp1 << std::endl;
    
    // Create b^2
    Value temp2(b.data * b.data, {&b}, {2*b.data});
    std::cout << "temp2 = b^2 = " << temp2.data << " at " << &temp2 << std::endl;
    
    // Create temp1 + temp2
    Value c(temp1.data + temp2.data, {&temp1, &temp2}, {1.0, 1.0});
    std::cout << "c = temp1 + temp2 = " << c.data << " at " << &c << std::endl;
    
    c.backward();
    
    std::cout << "After backward:" << std::endl;
    std::cout << "temp1.grad = " << temp1.grad << std::endl;
    std::cout << "temp2.grad = " << temp2.grad << std::endl;
    std::cout << "a.grad = " << a.grad << std::endl;
    std::cout << "b.grad = " << b.grad << std::endl;
    
    return 0;
}
