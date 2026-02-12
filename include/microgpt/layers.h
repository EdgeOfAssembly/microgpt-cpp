#pragma once

/**
 * Neural network layers (RMSNorm, Linear) - based on Andrej Karpathy's microGPT
 * Original Python implementation: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include "value.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace microgpt {

/**
 * RMS normalization - returns pointers to avoid copying
 * Includes safety checks for division by zero
 * Uses storage factory methods to eliminate stack temporaries
 */
inline std::vector<Value*> rmsnorm(const std::vector<Value*>& x, ValueStorage& storage) {
    assert(!x.empty() && "RMSNorm called with empty input");
    
    // Check for null pointers and validate data
    for (auto* xi : x) {
        assert(xi != nullptr && "Null pointer in rmsnorm input");
        assert(std::isfinite(xi->data) && "NaN or infinity in rmsnorm input");
    }
    
    Value* ms = storage.constant(0.0);
    for (auto* xi : x) {
        Value* sq = storage.mul(xi, xi);
        ms = storage.add(ms, sq);
    }
    
    // Check size to prevent overflow in static_cast
    if (x.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Vector size too large for rmsnorm");
    }
    
    Value* size_val = storage.constant(static_cast<double>(x.size()));
    ms = storage.div(ms, size_val);
    Value* eps_val = storage.constant(1e-5);
    Value* ms_eps = storage.add(ms, eps_val);
    
    // Validate before pow
    if (ms_eps->data <= 0.0) {
        throw std::domain_error("RMSNorm: mean square + epsilon is non-positive");
    }
    
    Value* scale = storage.pow(ms_eps, -0.5);
    
    // Check scale is reasonable
    if (!std::isfinite(scale->data) || std::abs(scale->data) > 1e10) {
        throw std::runtime_error("RMSNorm scale is invalid or too large");
    }

    std::vector<Value*> result;
    result.reserve(x.size());
    for (auto* xi : x) {
        result.push_back(storage.mul(xi, scale));
    }
    return result;
}

/**
 * Linear layer (matrix-vector multiplication) - returns pointers to avoid copying
 * Includes bounds checking and overflow detection
 * Uses storage factory methods to eliminate stack temporaries
 */
inline std::vector<Value*> linear(const std::vector<Value*>& x,
                                   std::vector<std::vector<Value>>& w,
                                   ValueStorage& storage) {
    assert(!x.empty() && "Linear called with empty input");
    assert(!w.empty() && "Linear called with empty weight matrix");
    
    // Validate dimensions
    for (const auto& wo : w) {
        if (wo.size() != x.size()) {
            throw std::invalid_argument("Linear: weight matrix dimensions don't match input");
        }
    }
    
    // Check for null pointers in input
    for (const auto* xi : x) {
        assert(xi != nullptr && "Null pointer in linear input");
        assert(std::isfinite(xi->data) && "NaN or infinity in linear input");
    }
    
    std::vector<Value*> result;
    result.reserve(w.size());
    
    for (auto& wo : w) {
        Value* sum = storage.constant(0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            // Validate weight
            assert(std::isfinite(wo[i].data) && "NaN or infinity in weight matrix");
            
            Value* prod = storage.mul(&wo[i], x[i]);
            sum = storage.add(sum, prod);
        }
        
        // Check for numerical issues
        if (!std::isfinite(sum->data)) {
            throw std::runtime_error("Linear layer produced NaN or infinity");
        }
        
        result.push_back(sum);
    }
    
    return result;
}

} // namespace microgpt
