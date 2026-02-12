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
 * Softmax function for Value vectors
 */
inline std::vector<Value> softmax(const std::vector<Value>& logits, ValueStorage& storage) {
    // Find max value for numerical stability
    double max_val = logits[0].data;
    for (const auto& val : logits) {
        if (val.data > max_val) {
            max_val = val.data;
        }
    }

    // Compute exp(x - max) and sum
    std::vector<Value*> exps;
    exps.reserve(logits.size());
    Value* total = storage.store(Value(0.0));
    
    for (const auto& val : logits) {
        Value* exp_val = storage.store(val - max_val);
        exp_val = storage.store(exp_val->exp());
        exps.push_back(exp_val);
        total = storage.store(*total + *exp_val);
    }

    // Normalize
    std::vector<Value> probs;
    probs.reserve(exps.size());
    for (auto* e : exps) {
        probs.push_back(*storage.store(*e / *total));
    }
    return probs;
}

/**
 * RMS normalization
 */
inline std::vector<Value> rmsnorm(const std::vector<Value>& x, ValueStorage& storage) {
    Value* ms = storage.store(Value(0.0));
    for (const auto& xi : x) {
        Value* sq = storage.store(xi * xi);
        ms = storage.store(*ms + *sq);
    }
    ms = storage.store(*ms / static_cast<double>(x.size()));
    Value* scale = storage.store((*ms + 1e-5).pow(-0.5));

    std::vector<Value> result;
    result.reserve(x.size());
    for (const auto& xi : x) {
        result.push_back(*storage.store(xi * *scale));
    }
    return result;
}

/**
 * Linear layer (matrix-vector multiplication)
 */
inline std::vector<Value> linear(const std::vector<Value>& x,
                                  const std::vector<std::vector<Value>>& w,
                                  ValueStorage& storage) {
    std::vector<Value> result;
    result.reserve(w.size());
    for (const auto& wo : w) {
        Value* sum = storage.store(Value(0.0));
        for (size_t i = 0; i < x.size(); ++i) {
            Value* prod = storage.store(wo[i] * x[i]);
            sum = storage.store(*sum + *prod);
        }
        result.push_back(*sum);
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
