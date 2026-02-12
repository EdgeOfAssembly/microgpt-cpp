#pragma once

#include "utils.h"
#include "value.h"
#include <map>
#include <random>
#include <string>
#include <vector>

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
                    params.push_back(&val);
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
 * GPT model
 */
class GPT {
public:
    Config config;
    StateDict state_dict;

    GPT(const Config& cfg) : config(cfg) {
        state_dict.init(config);
    }

    /**
     * Forward pass through the model
     * @param token_id Current token ID
     * @param pos_id Position ID
     * @param keys KV cache for keys
     * @param values KV cache for values
     * @param storage Value storage for intermediate computations
     * @return logits over vocabulary
     */
    std::vector<Value> forward(int token_id, int pos_id,
                                std::vector<std::vector<std::vector<Value>>>& keys,
                                std::vector<std::vector<std::vector<Value>>>& values,
                                ValueStorage& storage) {
        int head_dim = config.n_embd / config.n_head;

        // Token and position embeddings
        auto& tok_emb = state_dict.weights["wte"][token_id];
        auto& pos_emb = state_dict.weights["wpe"][pos_id];

        // Joint embedding
        std::vector<Value> x;
        x.reserve(config.n_embd);
        for (int i = 0; i < config.n_embd; ++i) {
            x.push_back(*storage.store(tok_emb[i] + pos_emb[i]));
        }
        x = rmsnorm(x, storage);

        // Transformer layers
        for (int li = 0; li < config.n_layer; ++li) {
            std::string prefix = "layer" + std::to_string(li) + ".";

            // 1) Multi-head attention
            auto x_residual = x;
            x = rmsnorm(x, storage);

            auto q = linear(x, state_dict.weights[prefix + "attn_wq"], storage);
            auto k = linear(x, state_dict.weights[prefix + "attn_wk"], storage);
            auto v = linear(x, state_dict.weights[prefix + "attn_wv"], storage);

            keys[li].push_back(k);
            values[li].push_back(v);

            std::vector<Value> x_attn;
            for (int h = 0; h < config.n_head; ++h) {
                int hs = h * head_dim;

                // Extract head-specific q, k, v
                std::vector<Value> q_h(q.begin() + hs, q.begin() + hs + head_dim);

                std::vector<std::vector<Value>> k_h;
                std::vector<std::vector<Value>> v_h;
                for (const auto& ki : keys[li]) {
                    k_h.push_back(std::vector<Value>(ki.begin() + hs, ki.begin() + hs + head_dim));
                }
                for (const auto& vi : values[li]) {
                    v_h.push_back(std::vector<Value>(vi.begin() + hs, vi.begin() + hs + head_dim));
                }

                // Compute attention scores
                std::vector<Value> attn_logits;
                double scale = std::sqrt(static_cast<double>(head_dim));
                for (size_t t = 0; t < k_h.size(); ++t) {
                    Value* score = storage.store(Value(0.0));
                    for (int j = 0; j < head_dim; ++j) {
                        Value* prod = storage.store(q_h[j] * k_h[t][j]);
                        score = storage.store(*score + *prod);
                    }
                    attn_logits.push_back(*storage.store(*score / scale));
                }

                // Softmax attention weights
                auto attn_weights = softmax(attn_logits, storage);

                // Weighted sum of values
                for (int j = 0; j < head_dim; ++j) {
                    Value* head_out = storage.store(Value(0.0));
                    for (size_t t = 0; t < v_h.size(); ++t) {
                        Value* prod = storage.store(attn_weights[t] * v_h[t][j]);
                        head_out = storage.store(*head_out + *prod);
                    }
                    x_attn.push_back(*head_out);
                }
            }

            x = linear(x_attn, state_dict.weights[prefix + "attn_wo"], storage);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = *storage.store(x[i] + x_residual[i]);
            }

            // 2) MLP block
            x_residual = x;
            x = rmsnorm(x, storage);
            x = linear(x, state_dict.weights[prefix + "mlp_fc1"], storage);
            for (auto& xi : x) {
                Value* relu_val = storage.store(xi.relu());
                xi = *storage.store(relu_val->pow(2));  // ReLU^2 activation
            }
            x = linear(x, state_dict.weights[prefix + "mlp_fc2"], storage);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = *storage.store(x[i] + x_residual[i]);
            }
        }

        // Final projection to logits
        auto logits = linear(x, state_dict.weights["lm_head"], storage);
        return logits;
    }

    /**
     * Generate text
     * @param start_token Starting token ID (usually BOS)
     * @param max_length Maximum generation length
     * @param temperature Sampling temperature
     * @return Generated token IDs
     */
    std::vector<int> generate(int start_token, int max_length, double temperature = 1.0) {
        ValueStorage storage;  // Local storage for generation
        std::vector<std::vector<std::vector<Value>>> keys(config.n_layer);
        std::vector<std::vector<std::vector<Value>>> values(config.n_layer);

        std::vector<int> tokens;
        int token_id = start_token;

        for (int pos_id = 0; pos_id < max_length && pos_id < config.block_size; ++pos_id) {
            auto logits = forward(token_id, pos_id, keys, values, storage);

            // Apply temperature
            std::vector<Value> scaled_logits;
            for (const auto& l : logits) {
                scaled_logits.push_back(*storage.store(l / temperature));
            }

            auto probs = softmax(scaled_logits, storage);

            // Sample from probability distribution
            std::vector<double> probs_data;
            for (const auto& p : probs) {
                probs_data.push_back(p.data);
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
