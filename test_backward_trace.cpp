#include <iostream>
#include <vector>
#include <deque>
#include <set>
#include <algorithm>
#include <cmath>

// Simplified Value for debugging
class Value {
public:
    double data, grad;
    std::vector<Value*> children_;
    std::vector<double> local_grads_;
    int id;  // For debugging
    static int next_id;
    
    Value(double d) : data(d), grad(0), id(next_id++) {
        std::cout << "Created Value " << id << " at " << this << std::endl;
    }
    
    Value(double d, std::vector<Value*> c, std::vector<double> lg) 
        : data(d), grad(0), children_(c), local_grads_(lg), id(next_id++) {
        std::cout << "Created Value " << id << " at " << this << " with " << c.size() << " children" << std::endl;
    }
    
    void backward() {
        std::cout << "backward() called on Value " << id << std::endl;
        std::vector<Value*> topo;
        std::set<Value*> visited;
        build_topo(this, topo, visited);
        std::cout << "Topo sort done, " << topo.size() << " nodes" << std::endl;
        
        grad = 1.0;
        std::reverse(topo.begin(), topo.end());
        for (Value* v : topo) {
            std::cout << "Processing Value " << v->id << " at " << v << std::endl;
            for (size_t i = 0; i < v->children_.size(); ++i) {
                std::cout << "  Child " << i << " at " << v->children_[i] << std::endl;
                std::cout << "  Accessing child->id..." << std::endl;
                std::cout << "  Child ID: " << v->children_[i]->id << std::endl;
                v->children_[i]->grad += v->local_grads_[i] * v->grad;
            }
        }
        std::cout << "backward() complete" << std::endl;
    }
    
private:
    void build_topo(Value* v, std::vector<Value*>& topo, std::set<Value*>& visited) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (Value* child : v->children_) {
                build_topo(child, topo, visited);
            }
            topo.push_back(v);
        }
    }
};

int Value::next_id = 0;

class ValueStorage {
public:
    std::deque<Value> values;
    
    Value* store(Value&& v) {
        values.push_back(std::move(v));
        std::cout << "Stored Value, now at " << &values.back() << std::endl;
        return &values.back();
    }
};

int main() {
    ValueStorage storage;
    
    Value* a = storage.store(Value(1.0));
    Value* b = storage.store(Value(2.0));
    
    // a + b
    Value* c = storage.store(Value(a->data + b->data, {a, b}, {1.0, 1.0}));
    
    std::cout << "\nCalling backward on c..." << std::endl;
    c->backward();
    
    std::cout << "\na->grad = " << a->grad << std::endl;
    std::cout << "b->grad = " << b->grad << std::endl;
    
    return 0;
}
