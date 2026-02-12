#pragma once

#include "value.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace microgpt {

/**
 * Simple character-level tokenizer
 */
class Tokenizer {
public:
    std::vector<char> uchars;  // unique characters
    int BOS;                    // Beginning of Sequence token
    int vocab_size;

    Tokenizer() = default;

    void fit(const std::vector<std::string>& docs) {
        std::set<char> char_set;
        for (const auto& doc : docs) {
            for (char c : doc) {
                char_set.insert(c);
            }
        }
        uchars = std::vector<char>(char_set.begin(), char_set.end());
        std::sort(uchars.begin(), uchars.end());
        BOS = static_cast<int>(uchars.size());
        vocab_size = static_cast<int>(uchars.size()) + 1;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens = {BOS};
        for (char c : text) {
            auto it = std::find(uchars.begin(), uchars.end(), c);
            if (it != uchars.end()) {
                tokens.push_back(static_cast<int>(it - uchars.begin()));
            }
        }
        tokens.push_back(BOS);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string text;
        for (int token : tokens) {
            if (token != BOS && token < static_cast<int>(uchars.size())) {
                text += uchars[token];
            }
        }
        return text;
    }
};

/**
 * Load documents from a file
 */
inline std::vector<std::string> load_docs(const std::string& filename) {
    std::vector<std::string> docs;
    std::ifstream file(filename);
    if (!file.is_open()) {
        return docs;
    }
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
            docs.push_back(line);
        }
    }
    return docs;
}

/**
 * Softmax function for Value vectors - returns pointers to avoid copying
 * All intermediate values are stored to ensure proper gradient flow
 * Includes safety checks for numerical stability
 * 
 * CRITICAL: Uses storage factory methods to eliminate stack temporaries
 */
inline std::vector<Value*> softmax(const std::vector<Value*>& logits, ValueStorage& storage) {
    assert(!logits.empty() && "Softmax called with empty logits");
    
    // Find max value for numerical stability
    double max_val = logits[0]->data;
    for (auto* val : logits) {
        assert(val != nullptr && "Null pointer in logits");
        assert(std::isfinite(val->data) && "NaN or infinity in logits");
        if (val->data > max_val) {
            max_val = val->data;
        }
    }

    // Store max_val as a Value so it can be part of the computation graph
    Value* max_val_node = storage.constant(max_val);

    // Compute exp(x - max) and sum
    std::vector<Value*> exps;
    exps.reserve(logits.size());
    Value* total = storage.constant(0.0);
    
    for (auto* val : logits) {
        // Use factory methods - no stack temporaries!
        Value* diff = storage.sub(val, max_val_node);
        Value* exp_val = storage.exp(diff);
        exps.push_back(exp_val);
        total = storage.add(total, exp_val);
    }

    // Check for numerical issues
    if (total->data < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Softmax normalization term too small (numerical instability)");
    }

    // Normalize - use factory methods
    std::vector<Value*> probs;
    probs.reserve(exps.size());
    Value* total_inv = storage.pow(total, -1.0);
    
    double prob_sum = 0.0;
    for (auto* e : exps) {
        Value* prob = storage.mul(e, total_inv);
        probs.push_back(prob);
        prob_sum += prob->data;
    }
    
    // Verify probabilities sum to 1 (within tolerance)
    assert(std::abs(prob_sum - 1.0) < 1e-6 && "Softmax probabilities don't sum to 1");
    
    return probs;
}

/**
 * RMS normalization - returns pointers to avoid copying
 * Includes safety checks for division by zero
 * Uses storage factory methods to eliminate stack temporaries
 */
inline std::vector<Value*> rmsnorm(const std::vector<Value*>& x, ValueStorage& storage) {
    assert(!x.empty() && "RMSNorm called with empty input");
    
    // Check for null pointers and validate data
    for (const auto* xi : x) {
        assert(xi != nullptr && "Null pointer in rmsnorm input");
        assert(std::isfinite(xi->data) && "NaN or infinity in rmsnorm input");
    }
    
    Value* ms = storage.constant(0.0);
    for (const auto* xi : x) {
        Value* sq = storage.mul(const_cast<Value*>(xi), const_cast<Value*>(xi));
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
    for (const auto* xi : x) {
        result.push_back(storage.mul(const_cast<Value*>(xi), scale));
    }
    return result;
}

/**
 * Linear layer (matrix-vector multiplication) - returns pointers to avoid copying
 * Includes bounds checking and overflow detection
 * Uses storage factory methods to eliminate stack temporaries
 */
inline std::vector<Value*> linear(const std::vector<Value*>& x,
                                   const std::vector<std::vector<Value>>& w,
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
    
    for (const auto& wo : w) {
        Value* sum = storage.constant(0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            // Validate weight
            assert(std::isfinite(wo[i].data) && "NaN or infinity in weight matrix");
            
            Value* prod = storage.mul(const_cast<Value*>(&wo[i]), x[i]);
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

/**
 * Random number generator (shared)
 */
inline std::mt19937& get_rng() {
    static std::mt19937 rng(42);
    return rng;
}

/**
 * Multinomial sampling from probability distribution
 */
inline int sample_multinomial(const std::vector<double>& probs) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(get_rng());
}

/**
 * Shuffle a vector
 */
template <typename T>
inline void shuffle(std::vector<T>& vec) {
    std::shuffle(vec.begin(), vec.end(), get_rng());
}

}  // namespace microgpt
