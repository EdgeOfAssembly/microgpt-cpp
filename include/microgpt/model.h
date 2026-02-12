#pragma once

/**
 * GPT model implementation - based on Andrej Karpathy's microGPT
 * Original Python implementation: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include "utils.h"
#include "value.h"
#include "optimizer.h"
#include <map>
#include <random>
#include <string>
#include <vector>
#include <fstream>

namespace microgpt {

/**
 * Model configuration
 */
struct Config {
    int vocab_size;
    int n_embd;
    int n_head;
    int n_layer;
    int block_size;
};

/**
 * Model state dictionary - stores all parameters
 */
class StateDict {
public:
    std::map<std::string, std::vector<std::vector<Value>>> weights;

    void init(const Config& config) {
        std::normal_distribution<double> dist(0.0, 0.02);
        std::normal_distribution<double> dist_zero(0.0, 0.0);
        auto& rng = get_rng();

        // Token and position embeddings
        weights["wte"] = create_matrix(config.vocab_size, config.n_embd, dist, rng);
        weights["wpe"] = create_matrix(config.block_size, config.n_embd, dist, rng);
        weights["lm_head"] = create_matrix(config.vocab_size, config.n_embd, dist, rng);

        // Layer weights
        for (int i = 0; i < config.n_layer; ++i) {
            std::string prefix = "layer" + std::to_string(i) + ".";
            weights[prefix + "attn_wq"] = create_matrix(config.n_embd, config.n_embd, dist, rng);
            weights[prefix + "attn_wk"] = create_matrix(config.n_embd, config.n_embd, dist, rng);
            weights[prefix + "attn_wv"] = create_matrix(config.n_embd, config.n_embd, dist, rng);
            weights[prefix + "attn_wo"] = create_matrix(config.n_embd, config.n_embd, dist_zero, rng);
            weights[prefix + "mlp_fc1"] = create_matrix(4 * config.n_embd, config.n_embd, dist, rng);
            weights[prefix + "mlp_fc2"] = create_matrix(config.n_embd, 4 * config.n_embd, dist_zero, rng);
        }
    }

    std::vector<Value*> get_all_params() {
        std::vector<Value*> params;
        for (auto& [name, matrix] : weights) {
            for (auto& row : matrix) {
                for (auto& val : row) {
                    Value* ptr = &val;
                    assert(ptr != nullptr && "Parameter pointer is null");
                    params.push_back(ptr);
                }
            }
        }
        return params;
    }

    std::vector<const Value*> get_all_params() const {
        std::vector<const Value*> params;
        for (const auto& [name, matrix] : weights) {
            for (const auto& row : matrix) {
                for (const auto& val : row) {
                    const Value* ptr = &val;
                    assert(ptr != nullptr && "Parameter pointer is null");
                    params.push_back(ptr);
                }
            }
        }
        return params;
    }

private:
    template <typename Dist, typename RNG>
    std::vector<std::vector<Value>> create_matrix(int nout, int nin, Dist& dist, RNG& rng) {
        std::vector<std::vector<Value>> matrix;
        matrix.reserve(nout);
        for (int i = 0; i < nout; ++i) {
            std::vector<Value> row;
            row.reserve(nin);
            for (int j = 0; j < nin; ++j) {
                row.emplace_back(dist(rng));
            }
            matrix.push_back(std::move(row));
        }
        return matrix;
    }
};

/**
 * GPT model with simple educational API
 */
class GPT {
public:
    Config config;
    StateDict state_dict;

    GPT(const Config& cfg) : config(cfg) {
        state_dict.init(config);
    }

    /**
     * Save model weights and config to binary file
     */
    void save_weights(const std::string& filename, const Tokenizer& tokenizer) const {
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        // Write config
        outfile.write(reinterpret_cast<const char*>(&config.vocab_size), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.n_embd), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.n_head), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.n_layer), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&config.block_size), sizeof(int));

        // Write tokenizer
        int uchars_size = static_cast<int>(tokenizer.uchars.size());
        outfile.write(reinterpret_cast<const char*>(&uchars_size), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(tokenizer.uchars.data()), uchars_size);
        outfile.write(reinterpret_cast<const char*>(&tokenizer.BOS), sizeof(int));

        // Write parameters
        auto params = state_dict.get_all_params();
        for (const auto* p : params) {
            outfile.write(reinterpret_cast<const char*>(&p->data), sizeof(double));
        }

        if (!outfile) {
            throw std::runtime_error("Error writing to file: " + filename);
        }
    }

    /**
     * Load model weights and config from binary file
     * Returns the loaded model and tokenizer
     */
    static std::pair<GPT, Tokenizer> load_weights(const std::string& filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }

        // Read config
        Config config{};
        infile.read(reinterpret_cast<char*>(&config.vocab_size), sizeof(int));
        infile.read(reinterpret_cast<char*>(&config.n_embd), sizeof(int));
        infile.read(reinterpret_cast<char*>(&config.n_head), sizeof(int));
        infile.read(reinterpret_cast<char*>(&config.n_layer), sizeof(int));
        infile.read(reinterpret_cast<char*>(&config.block_size), sizeof(int));
        
        if (!infile) {
            throw std::runtime_error("Failed to read config from file");
        }

        // Validate config
        if (config.vocab_size <= 0 || config.n_embd <= 0 || config.n_head <= 0 ||
            config.n_layer <= 0 || config.block_size <= 0) {
            throw std::runtime_error("Invalid config in file");
        }
        if (config.n_embd % config.n_head != 0) {
            throw std::runtime_error("n_embd must be divisible by n_head");
        }

        // Read tokenizer
        Tokenizer tokenizer;
        int uchars_size = 0;
        infile.read(reinterpret_cast<char*>(&uchars_size), sizeof(int));
        if (!infile) {
            throw std::runtime_error("Failed to read tokenizer size from file");
        }
        if (uchars_size <= 0 || uchars_size > 10000) {
            throw std::runtime_error("Invalid tokenizer size in file");
        }
        tokenizer.uchars.resize(uchars_size);
        infile.read(reinterpret_cast<char*>(tokenizer.uchars.data()), uchars_size);
        if (!infile) {
            throw std::runtime_error("Failed to read tokenizer characters from file");
        }
        infile.read(reinterpret_cast<char*>(&tokenizer.BOS), sizeof(int));
        if (!infile) {
            throw std::runtime_error("Failed to read tokenizer BOS from file");
        }
        tokenizer.vocab_size = config.vocab_size;
        
        // Validate tokenizer consistency
        if (tokenizer.BOS != uchars_size) {
            throw std::runtime_error("Inconsistent tokenizer BOS index in file");
        }
        if (config.vocab_size != uchars_size + 1) {
            throw std::runtime_error("Incompatible vocab_size between config and tokenizer");
        }

        // Initialize model
        GPT model(config);
        auto params = model.state_dict.get_all_params();

        // Load parameters
        for (auto* p : params) {
            infile.read(reinterpret_cast<char*>(&p->data), sizeof(double));
            if (!std::isfinite(p->data)) {
                throw std::runtime_error("Loaded parameter with NaN or infinity");
            }
        }

        if (!infile) {
            throw std::runtime_error("Failed to read all parameters from file");
        }

        return {model, tokenizer};
    }

    /**
     * Single training step on a sequence
     * Returns the loss value
     */
    double train_step(const std::vector<int>& tokens, Adam& optimizer, ValueStorage& storage, int total_steps) {
        const int n = std::min(config.block_size, static_cast<int>(tokens.size()) - 1);
        if (n <= 0) {
            return 0.0;  // Skip empty sequences
        }

        // Forward pass
        std::vector<std::vector<std::vector<Value*>>> keys(config.n_layer);
        std::vector<std::vector<std::vector<Value*>>> values(config.n_layer);
        std::vector<Value*> losses;
        losses.reserve(n);

        for (int pos_id = 0; pos_id < n; ++pos_id) {
            const int token_id = tokens[pos_id];
            const int target_id = tokens[pos_id + 1];

            auto logits = forward(token_id, pos_id, keys, values, storage);
            auto probs = softmax(logits, storage);

            Value* log_prob = storage.log(probs[target_id]);
            Value* loss_t = storage.neg(log_prob);
            losses.push_back(loss_t);
        }

        // Average loss
        Value* loss = storage.constant(0.0);
        for (auto* l : losses) {
            loss = storage.add(loss, l);
        }
        Value* n_val = storage.constant(static_cast<double>(n));
        loss = storage.div(loss, n_val);

        // Backward pass
        loss->backward();

        // Optimizer step
        auto params = state_dict.get_all_params();
        optimizer.step(params, total_steps);

        return loss->data;
    }

    /**
     * Forward pass through the model - uses const references and pointers to avoid copying
     * Includes comprehensive safety checks
     * @param token_id Current token ID
     * @param pos_id Position ID
     * @param keys KV cache for keys
     * @param values KV cache for values
     * @param storage Value storage for intermediate computations
     * @return logits over vocabulary (as pointers)
     */
    std::vector<Value*> forward(int token_id, int pos_id,
                                 std::vector<std::vector<std::vector<Value*>>>& keys,
                                 std::vector<std::vector<std::vector<Value*>>>& values,
                                 ValueStorage& storage) {
        // Bounds checking
        if (token_id < 0 || token_id >= config.vocab_size) {
            throw std::out_of_range("token_id out of range");
        }
        if (pos_id < 0 || pos_id >= config.block_size) {
            throw std::out_of_range("pos_id out of range");
        }
        
        // Check storage isn't growing too large (potential memory leak)
        storage.check_size_limit(100000);
        
        const int head_dim = config.n_embd / config.n_head;
        
        // Validate head_dim
        if (head_dim <= 0 || config.n_embd % config.n_head != 0) {
            throw std::invalid_argument("n_embd must be divisible by n_head");
        }

        // Token and position embeddings
        auto& tok_emb = state_dict.weights["wte"][token_id];
        auto& pos_emb = state_dict.weights["wpe"][pos_id];

        // Joint embedding - use factory methods
        std::vector<Value*> x;
        x.reserve(config.n_embd);
        for (int i = 0; i < config.n_embd; ++i) {
            x.push_back(storage.add(&tok_emb[i], &pos_emb[i]));
        }
        x = rmsnorm(x, storage);

        // Transformer layers
        for (int li = 0; li < config.n_layer; ++li) {
            const std::string prefix = "layer" + std::to_string(li) + ".";

            // 1) Multi-head attention
            auto x_residual = x;  // Copy pointers, not values
            x = rmsnorm(x, storage);

            auto q = linear(x, state_dict.weights[prefix + "attn_wq"], storage);
            auto k = linear(x, state_dict.weights[prefix + "attn_wk"], storage);
            auto v = linear(x, state_dict.weights[prefix + "attn_wv"], storage);

            keys[li].push_back(k);
            values[li].push_back(v);

            std::vector<Value*> x_attn;
            x_attn.reserve(config.n_embd);
            
            for (int h = 0; h < config.n_head; ++h) {
                const int hs = h * head_dim;
                
                // Bounds check for head slicing
                if (hs + head_dim > static_cast<int>(q.size())) {
                    throw std::out_of_range("Head slicing out of bounds");
                }

                // Extract head-specific q, k, v (copy pointers only)
                std::vector<Value*> q_h(q.begin() + hs, q.begin() + hs + head_dim);

                std::vector<std::vector<Value*>> k_h;
                std::vector<std::vector<Value*>> v_h;
                k_h.reserve(keys[li].size());
                v_h.reserve(values[li].size());
                
                for (const auto& ki : keys[li]) {
                    if (hs + head_dim > static_cast<int>(ki.size())) {
                        throw std::out_of_range("Key head slicing out of bounds");
                    }
                    k_h.emplace_back(ki.begin() + hs, ki.begin() + hs + head_dim);
                }
                for (const auto& vi : values[li]) {
                    if (hs + head_dim > static_cast<int>(vi.size())) {
                        throw std::out_of_range("Value head slicing out of bounds");
                    }
                    v_h.emplace_back(vi.begin() + hs, vi.begin() + hs + head_dim);
                }

                // Compute attention scores
                std::vector<Value*> attn_logits;
                attn_logits.reserve(k_h.size());
                const double scale = std::sqrt(static_cast<double>(head_dim));
                
                // Check scale is valid
                if (!std::isfinite(scale) || scale <= 0.0) {
                    throw std::runtime_error("Invalid attention scale");
                }
                
                for (size_t t = 0; t < k_h.size(); ++t) {
                    Value* score = storage.constant(0.0);
                    for (int j = 0; j < head_dim; ++j) {
                        assert(q_h[j] != nullptr && "Null pointer in q_h");
                        assert(k_h[t][j] != nullptr && "Null pointer in k_h");
                        Value* prod = storage.mul(q_h[j], k_h[t][j]);
                        score = storage.add(score, prod);
                    }
                    
                    // Use factory method for division
                    Value* scale_val = storage.constant(scale);
                    attn_logits.push_back(storage.div(score, scale_val));
                }

                // Softmax attention weights
                auto attn_weights = softmax(attn_logits, storage);

                // Weighted sum of values
                for (int j = 0; j < head_dim; ++j) {
                    Value* head_out = storage.constant(0.0);
                    for (size_t t = 0; t < v_h.size(); ++t) {
                        assert(attn_weights[t] != nullptr && "Null pointer in attn_weights");
                        assert(v_h[t][j] != nullptr && "Null pointer in v_h");
                        Value* prod = storage.mul(attn_weights[t], v_h[t][j]);
                        head_out = storage.add(head_out, prod);
                    }
                    x_attn.push_back(head_out);
                }
            }

            x = linear(x_attn, state_dict.weights[prefix + "attn_wo"], storage);
            
            // Validate dimensions match for residual
            if (x.size() != x_residual.size()) {
                throw std::runtime_error("Dimension mismatch in attention residual connection");
            }
            
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = storage.add(x[i], x_residual[i]);
            }

            // 2) MLP block
            x_residual = x;
            x = rmsnorm(x, storage);
            x = linear(x, state_dict.weights[prefix + "mlp_fc1"], storage);
            for (auto*& xi : x) {
                assert(xi != nullptr && "Null pointer in MLP activation");
                Value* relu_val = storage.relu(xi);
                xi = storage.pow(relu_val, 2.0);  // ReLU^2 activation
            }
            x = linear(x, state_dict.weights[prefix + "mlp_fc2"], storage);
            
            // Validate dimensions match for residual
            if (x.size() != x_residual.size()) {
                throw std::runtime_error("Dimension mismatch in MLP residual connection");
            }
            
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = storage.add(x[i], x_residual[i]);
            }
        }

        // Final projection to logits
        auto logits = linear(x, state_dict.weights["lm_head"], storage);
        
        // Validate output dimensions
        if (static_cast<int>(logits.size()) != config.vocab_size) {
            throw std::runtime_error("Output logits size doesn't match vocab_size");
        }
        
        return logits;
    }

    /**
     * Generate text - uses const references to avoid copying
     * @param start_token Starting token ID (usually BOS)
     * @param max_length Maximum generation length
     * @param temperature Sampling temperature
     * @return Generated token IDs
     */
    std::vector<int> generate(int start_token, int max_length, double temperature = 1.0) {
        ValueStorage storage;  // Local storage for generation
        std::vector<std::vector<std::vector<Value*>>> keys(config.n_layer);
        std::vector<std::vector<std::vector<Value*>>> values(config.n_layer);

        std::vector<int> tokens;
        tokens.reserve(max_length);
        int token_id = start_token;

        for (int pos_id = 0; pos_id < max_length && pos_id < config.block_size; ++pos_id) {
            auto logits = forward(token_id, pos_id, keys, values, storage);

            // Apply temperature using factory method
            std::vector<Value*> scaled_logits;
            scaled_logits.reserve(logits.size());
            for (auto* l : logits) {
                scaled_logits.push_back(storage.div(l, temperature));
            }

            auto probs = softmax(scaled_logits, storage);

            // Sample from probability distribution
            std::vector<double> probs_data;
            probs_data.reserve(probs.size());
            for (const auto* p : probs) {
                probs_data.push_back(p->data);
            }
            token_id = sample_multinomial(probs_data);

            if (token_id == start_token) {  // BOS token ends generation
                break;
            }
            tokens.push_back(token_id);
        }

        return tokens;
    }
};

}  // namespace microgpt
