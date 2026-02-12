#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace microgpt {

/**
 * Stores a single scalar value and its gradient, as a node in a computation graph.
 * Direct port of Python's Value class for scalar autograd.
 * 
 * IMPORTANT: All Value objects must outlive any backward() calls on values that depend on them.
 * The recommended pattern is to store all intermediate computation results in ValueStorage.
 */
class Value {
public:
    double data;  // scalar value calculated during forward pass
    double grad;  // derivative of loss w.r.t. this node, calculated in backward pass

    Value(double data = 0.0) : data(data), grad(0.0), children_(), local_grads_() {
        // Check for NaN or infinity
        assert(std::isfinite(data) && "Value initialized with NaN or infinity");
    }

    Value(double data, std::vector<Value*> children, std::vector<double> local_grads)
        : data(data), grad(0.0), children_(std::move(children)), local_grads_(std::move(local_grads)) {
        assert(std::isfinite(data) && "Value initialized with NaN or infinity");
        assert(children_.size() == local_grads_.size() && "Mismatched children and local_grads sizes");
        
        // Validate all children pointers are non-null
        for (const auto* child : children_) {
            assert(child != nullptr && "Null pointer in children");
        }
    }

    // ========================================================================
    // DEPRECATED: Direct operator overloads (create stack temporaries)
    // DO NOT USE THESE DIRECTLY - Use ValueStorage factory methods instead!
    // 
    // These operators return Values by value, creating stack-allocated 
    // temporaries that cause dangling pointer issues in computation graphs.
    // 
    // Instead of:  storage.store(*a + *b)
    // Use:         storage.add(a, b)
    // ========================================================================
    
    // Addition (DEPRECATED - use storage.add())
    [[deprecated("Use ValueStorage::add() to avoid stack temporaries")]]
    Value operator+(const Value& other) const {
        // Check for overflow
        if (data > 0 && other.data > 0 && data > std::numeric_limits<double>::max() - other.data) {
            throw std::overflow_error("Addition would overflow");
        }
        if (data < 0 && other.data < 0 && data < std::numeric_limits<double>::lowest() - other.data) {
            throw std::underflow_error("Addition would underflow");
        }
        return Value(data + other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, {1.0, 1.0});
    }

    [[deprecated("Use ValueStorage::add() to avoid stack temporaries")]]
    Value operator+(double other) const {
        assert(std::isfinite(other) && "Adding NaN or infinity");
        return Value(data + other, {const_cast<Value*>(this)}, {1.0});
    }

    [[deprecated("Use ValueStorage::add() to avoid stack temporaries")]]
    friend Value operator+(double lhs, const Value& rhs) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return rhs + lhs;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    // Multiplication (DEPRECATED - use storage.mul())
    [[deprecated("Use ValueStorage::mul() to avoid stack temporaries")]]
    Value operator*(const Value& other) const {
        const double result = data * other.data;
        assert(std::isfinite(result) && "Multiplication resulted in NaN or infinity");
        return Value(result, {const_cast<Value*>(this), const_cast<Value*>(&other)},
                     {other.data, data});
    }

    [[deprecated("Use ValueStorage::mul() to avoid stack temporaries")]]
    Value operator*(double other) const {
        assert(std::isfinite(other) && "Multiplying by NaN or infinity");
        const double result = data * other;
        assert(std::isfinite(result) && "Multiplication resulted in NaN or infinity");
        return Value(result, {const_cast<Value*>(this)}, {other});
    }

    [[deprecated("Use ValueStorage::mul() to avoid stack temporaries")]]
    friend Value operator*(double lhs, const Value& rhs) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return rhs * lhs;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    // Negation (DEPRECATED - use storage.neg())
    [[deprecated("Use ValueStorage::neg() to avoid stack temporaries")]]
    Value operator-() const {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return (*this) * -1.0;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    // Subtraction (DEPRECATED - use storage.sub())
    [[deprecated("Use ValueStorage::sub() to avoid stack temporaries")]]
    Value operator-(const Value& other) const {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return (*this) + (-other);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    [[deprecated("Use ValueStorage::sub() to avoid stack temporaries")]]
    Value operator-(double other) const {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return (*this) + (-other);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    [[deprecated("Use ValueStorage::sub() to avoid stack temporaries")]]
    friend Value operator-(double lhs, const Value& rhs) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return lhs + (-rhs);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    // Power (use storage.pow() instead)
    Value pow(double exponent) const {
        assert(std::isfinite(exponent) && "Power exponent is NaN or infinity");
        
        // Check for domain errors
        if (data < 0.0 && std::floor(exponent) != exponent) {
            throw std::domain_error("Negative base with non-integer exponent");
        }
        if (data == 0.0 && exponent < 0.0) {
            throw std::domain_error("Zero to negative power (division by zero)");
        }
        
        const double result = std::pow(data, exponent);
        assert(std::isfinite(result) && "Power resulted in NaN or infinity");
        
        const double local_grad = exponent * std::pow(data, exponent - 1);
        assert(std::isfinite(local_grad) && "Power gradient is NaN or infinity");
        
        return Value(result, {const_cast<Value*>(this)}, {local_grad});
    }

    // Division (DEPRECATED - use storage.div())
    [[deprecated("Use ValueStorage::div() to avoid stack temporaries")]]
    Value operator/(const Value& other) const {
        if (std::abs(other.data) < std::numeric_limits<double>::epsilon()) {
            throw std::domain_error("Division by zero or near-zero value");
        }
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return (*this) * other.pow(-1);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    [[deprecated("Use ValueStorage::div() to avoid stack temporaries")]]
    Value operator/(double other) const {
        if (std::abs(other) < std::numeric_limits<double>::epsilon()) {
            throw std::domain_error("Division by zero or near-zero value");
        }
        assert(std::isfinite(other) && "Dividing by NaN or infinity");
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return (*this) * (1.0 / other);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    [[deprecated("Use ValueStorage::div() to avoid stack temporaries")]]
    friend Value operator/(double lhs, const Value& rhs) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        return lhs * rhs.pow(-1);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    }

    // Mathematical functions
    Value log() const {
        if (data <= 0.0) {
            throw std::domain_error("Log of non-positive value");
        }
        const double result = std::log(data);
        assert(std::isfinite(result) && "Log resulted in NaN or infinity");
        return Value(result, {const_cast<Value*>(this)}, {1.0 / data});
    }

    Value exp() const {
        // Check for potential overflow
        if (data > 700.0) {  // exp(700) is near double max
            throw std::overflow_error("Exp would overflow");
        }
        const double result = std::exp(data);
        assert(std::isfinite(result) && "Exp resulted in NaN or infinity");
        return Value(result, {const_cast<Value*>(this)}, {result});
    }

    Value relu() const {
        const double result = std::max(0.0, data);
        const double local_grad = (data > 0) ? 1.0 : 0.0;
        return Value(result, {const_cast<Value*>(this)}, {local_grad});
    }

    // Backward pass with safety checks
    void backward() {
        std::vector<Value*> topo;
        std::set<Value*> visited;
        
        try {
            build_topo(this, topo, visited);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error building computation graph: ") + e.what());
        }

        grad = 1.0;
        std::reverse(topo.begin(), topo.end());
        
        for (Value* v : topo) {
            assert(v != nullptr && "Null pointer in topological order");
            
            // Validate this node hasn't been freed
            assert(std::isfinite(v->data) && "Node data is NaN or infinity (possible use-after-free)");
            assert(std::isfinite(v->grad) && "Node grad is NaN or infinity");
            
            for (size_t i = 0; i < v->children_.size(); ++i) {
                Value* child = v->children_[i];
                
                // Critical pointer validation
                if (child == nullptr) {
                    throw std::runtime_error("Null child pointer detected in computation graph");
                }
                
                // Validate gradient computation
                const double grad_contribution = v->local_grads_[i] * v->grad;
                assert(std::isfinite(grad_contribution) && "Gradient contribution is NaN or infinity");
                
                child->grad += grad_contribution;
                
                // Check for gradient explosion
                if (std::abs(child->grad) > 1e10) {
                    // Warning: gradient may be exploding (but continue)
                }
            }
        }
    }

private:
    std::vector<Value*> children_;
    std::vector<double> local_grads_;

    void build_topo(Value* v, std::vector<Value*>& topo, std::set<Value*>& visited) const {
        if (v == nullptr) {
            throw std::runtime_error("Null pointer in computation graph");
        }
        
        // Check for cycles (shouldn't happen in DAG, but detect infinite recursion)
        if (topo.size() > 100000) {
            throw std::runtime_error("Computation graph too large or has cycle");
        }
        
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            
            // Validate this node hasn't been corrupted
            assert(std::isfinite(v->data) && "Corrupted node in graph (NaN/inf data)");
            
            for (size_t i = 0; i < v->children_.size(); ++i) {
                Value* child = v->children_[i];
                if (child == nullptr) {
                    // Print debug info
                    std::cerr << "Null child detected:" << std::endl;
                    std::cerr << "  Parent Value at: " << v << std::endl;
                    std::cerr << "  Parent data: " << v->data << std::endl;
                    std::cerr << "  Child index: " << i << " of " << v->children_.size() << std::endl;
                    std::cerr << "  Local grad: " << (i < v->local_grads_.size() ? v->local_grads_[i] : -999.0) << std::endl;
                    throw std::runtime_error("Null child pointer in graph traversal");
                }
                build_topo(child, topo, visited);
            }
            topo.push_back(v);
        }
    }
};

/**
 * Helper to store intermediate Value computation nodes with safety checks
 * Use this to ensure all Values in a computation stay alive for backward pass
 * 
 * IMPORTANT: Always use factory methods (add, mul, etc.) instead of operators!
 * Factory methods ensure ALL Values are heap-allocated, preventing dangling pointers.
 * 
 * ✅ CORRECT:   storage.add(a, b)
 * ❌ WRONG:     storage.store(*a + *b)  // Creates stack temporary!
 * 
 * Factory methods that automatically store results:
 * - constant(double) - Create a constant Value
 * - add(a, b) - Addition
 * - sub(a, b) - Subtraction  
 * - mul(a, b) - Multiplication
 * - div(a, b) - Division
 * - neg(a) - Negation
 * - pow(a, exp) - Power
 * - log(a) - Natural logarithm
 * - exp(a) - Exponential
 * - relu(a) - ReLU activation
 */
class ValueStorage {
public:
    std::deque<Value> values;  // deque ensures pointers remain valid when growing
    
    Value* store(Value&& v) {
        // Validate the value before storing
        assert(std::isfinite(v.data) && "Attempting to store NaN or infinity");
        
        values.push_back(std::move(v));
        Value* ptr = &values.back();
        
        // Validate pointer is in valid range
        assert(ptr != nullptr && "Storage returned null pointer");
        
        return ptr;
    }
    
    // Factory method: Create a constant Value
    Value* constant(double data) {
        return store(Value(data));
    }
    
    // Factory method: Addition
    Value* add(Value* a, Value* b) {
        assert(a != nullptr && "Null pointer in add");
        assert(b != nullptr && "Null pointer in add");
        // Create Value directly without using operator overload
        double result = a->data + b->data;
        if (a->data > 0 && b->data > 0 && a->data > std::numeric_limits<double>::max() - b->data) {
            throw std::overflow_error("Addition would overflow");
        }
        return store(Value(result, {a, b}, {1.0, 1.0}));
    }
    
    Value* add(Value* a, double b) {
        assert(a != nullptr && "Null pointer in add");
        assert(std::isfinite(b) && "Adding NaN or infinity");
        return store(Value(a->data + b, {a}, {1.0}));
    }
    
    // Factory method: Multiplication
    Value* mul(Value* a, Value* b) {
        assert(a != nullptr && "Null pointer in mul");
        assert(b != nullptr && "Null pointer in mul");
        const double result = a->data * b->data;
        assert(std::isfinite(result) && "Multiplication resulted in NaN or infinity");
        return store(Value(result, {a, b}, {b->data, a->data}));
    }
    
    Value* mul(Value* a, double b) {
        assert(a != nullptr && "Null pointer in mul");
        assert(std::isfinite(b) && "Multiplying by NaN or infinity");
        const double result = a->data * b;
        assert(std::isfinite(result) && "Multiplication resulted in NaN or infinity");
        return store(Value(result, {a}, {b}));
    }
    
    // Factory method: Negation
    Value* neg(Value* a) {
        assert(a != nullptr && "Null pointer in neg");
        return mul(a, -1.0);
    }
    
    // Factory method: Subtraction
    Value* sub(Value* a, Value* b) {
        assert(a != nullptr && "Null pointer in sub");
        assert(b != nullptr && "Null pointer in sub");
        Value* neg_b = neg(b);
        return add(a, neg_b);
    }
    
    Value* sub(Value* a, double b) {
        assert(a != nullptr && "Null pointer in sub");
        return add(a, -b);
    }
    
    // Factory method: Power
    Value* pow(Value* a, double exponent) {
        assert(a != nullptr && "Null pointer in pow");
        assert(std::isfinite(exponent) && "Power exponent is NaN or infinity");
        
        if (a->data < 0.0 && std::floor(exponent) != exponent) {
            throw std::domain_error("Negative base with non-integer exponent");
        }
        if (a->data == 0.0 && exponent < 0.0) {
            throw std::domain_error("Zero to negative power (division by zero)");
        }
        
        const double result = std::pow(a->data, exponent);
        assert(std::isfinite(result) && "Power resulted in NaN or infinity");
        
        const double local_grad = exponent * std::pow(a->data, exponent - 1);
        assert(std::isfinite(local_grad) && "Power gradient is NaN or infinity");
        
        return store(Value(result, {a}, {local_grad}));
    }
    
    // Factory method: Division
    Value* div(Value* a, Value* b) {
        assert(a != nullptr && "Null pointer in div");
        assert(b != nullptr && "Null pointer in div");
        if (std::abs(b->data) < std::numeric_limits<double>::epsilon()) {
            throw std::domain_error("Division by zero or near-zero value");
        }
        Value* b_inv = pow(b, -1.0);
        return mul(a, b_inv);
    }
    
    Value* div(Value* a, double b) {
        assert(a != nullptr && "Null pointer in div");
        if (std::abs(b) < std::numeric_limits<double>::epsilon()) {
            throw std::domain_error("Division by zero or near-zero value");
        }
        assert(std::isfinite(b) && "Dividing by NaN or infinity");
        return mul(a, 1.0 / b);
    }
    
    // Factory method: Logarithm
    Value* log(Value* a) {
        assert(a != nullptr && "Null pointer in log");
        if (a->data <= 0.0) {
            throw std::domain_error("Log of non-positive value");
        }
        const double result = std::log(a->data);
        assert(std::isfinite(result) && "Log resulted in NaN or infinity");
        return store(Value(result, {a}, {1.0 / a->data}));
    }
    
    // Factory method: Exponential
    Value* exp(Value* a) {
        assert(a != nullptr && "Null pointer in exp");
        if (a->data > 700.0) {
            throw std::overflow_error("Exp would overflow");
        }
        const double result = std::exp(a->data);
        assert(std::isfinite(result) && "Exp resulted in NaN or infinity");
        return store(Value(result, {a}, {result}));
    }
    
    // Factory method: ReLU
    Value* relu(Value* a) {
        assert(a != nullptr && "Null pointer in relu");
        const double result = std::max(0.0, a->data);
        const double local_grad = (a->data > 0) ? 1.0 : 0.0;
        return store(Value(result, {a}, {local_grad}));
    }
    
    void clear() {
        values.clear();
    }
    
    size_t size() const {
        return values.size();
    }
    
    // Check for memory usage growth
    void check_size_limit(size_t max_size = 1000000) const {
        if (values.size() > max_size) {
            throw std::runtime_error("ValueStorage exceeded size limit - possible memory leak");
        }
    }
};

}  // namespace microgpt
