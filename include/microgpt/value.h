#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <set>
#include <vector>

namespace microgpt {

/**
 * Stores a single scalar value and its gradient, as a node in a computation graph.
 * Direct port of Python's Value class for scalar autograd.
 * 
 * IMPORTANT: All Value objects must outlive any backward() calls on values that depend on them.
 * The recommended pattern is to store all intermediate computation results in vectors.
 */
class Value {
public:
    double data;  // scalar value calculated during forward pass
    double grad;  // derivative of loss w.r.t. this node, calculated in backward pass

    Value(double data = 0.0) : data(data), grad(0.0), children_(), local_grads_() {}

    Value(double data, std::vector<Value*> children, std::vector<double> local_grads)
        : data(data), grad(0.0), children_(std::move(children)), local_grads_(std::move(local_grads)) {}

    // Addition
    Value operator+(const Value& other) const {
        return Value(data + other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, {1.0, 1.0});
    }

    Value operator+(double other) const {
        return Value(data + other, {const_cast<Value*>(this)}, {1.0});
    }

    friend Value operator+(double lhs, const Value& rhs) {
        return rhs + lhs;
    }

    // Multiplication
    Value operator*(const Value& other) const {
        return Value(data * other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)},
                     {other.data, data});
    }

    Value operator*(double other) const {
        return Value(data * other, {const_cast<Value*>(this)}, {other});
    }

    friend Value operator*(double lhs, const Value& rhs) {
        return rhs * lhs;
    }

    // Negation
    Value operator-() const { return (*this) * -1.0; }

    // Subtraction
    Value operator-(const Value& other) const { return (*this) + (-other); }

    Value operator-(double other) const { return (*this) + (-other); }

    friend Value operator-(double lhs, const Value& rhs) {
        return lhs + (-rhs);
    }

    // Power
    Value pow(double exponent) const {
        double result = std::pow(data, exponent);
        double local_grad = exponent * std::pow(data, exponent - 1);
        return Value(result, {const_cast<Value*>(this)}, {local_grad});
    }

    // Division
    Value operator/(const Value& other) const {
        return (*this) * other.pow(-1);
    }

    Value operator/(double other) const {
        return (*this) * (1.0 / other);
    }

    friend Value operator/(double lhs, const Value& rhs) {
        return lhs * rhs.pow(-1);
    }

    // Mathematical functions
    Value log() const {
        return Value(std::log(data), {const_cast<Value*>(this)}, {1.0 / data});
    }

    Value exp() const {
        double result = std::exp(data);
        return Value(result, {const_cast<Value*>(this)}, {result});
    }

    Value relu() const {
        double result = std::max(0.0, data);
        double local_grad = (data > 0) ? 1.0 : 0.0;
        return Value(result, {const_cast<Value*>(this)}, {local_grad});
    }

    // Backward pass
    void backward() {
        std::vector<Value*> topo;
        std::set<Value*> visited;
        build_topo(this, topo, visited);

        grad = 1.0;
        std::reverse(topo.begin(), topo.end());
        for (Value* v : topo) {
            for (size_t i = 0; i < v->children_.size(); ++i) {
                v->children_[i]->grad += v->local_grads_[i] * v->grad;
            }
        }
    }

private:
    std::vector<Value*> children_;
    std::vector<double> local_grads_;

    void build_topo(Value* v, std::vector<Value*>& topo, std::set<Value*>& visited) const {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (Value* child : v->children_) {
                build_topo(child, topo, visited);
            }
            topo.push_back(v);
        }
    }
};

/**
 * Helper to store intermediate Value computation nodes
 * Use this to ensure all Values in a computation stay alive for backward pass
 */
class ValueStorage {
public:
    std::deque<Value> values;  // deque ensures pointers remain valid when growing
    
    Value* store(Value&& v) {
        values.push_back(std::move(v));
        return &values.back();
    }
    
    void clear() {
        values.clear();
    }
};

}  // namespace microgpt
