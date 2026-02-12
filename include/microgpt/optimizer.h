#pragma once

/**
 * Adam optimizer - based on Andrej Karpathy's microGPT
 * Original Python implementation: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include "value.h"
#include <cmath>
#include <numbers>
#include <vector>

namespace microgpt {

/**
 * Adam optimizer with bias correction and cosine learning rate schedule
 */
class Adam {
public:
    double learning_rate;
    double beta1;
    double beta2;
    double eps;
    int step_count;

    std::vector<double> m;  // first moment buffer
    std::vector<double> v;  // second moment buffer

    Adam(double lr = 1e-2, double b1 = 0.9, double b2 = 0.95, double epsilon = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), eps(epsilon), step_count(0) {}

    void init(size_t num_params) {
        m.resize(num_params, 0.0);
        v.resize(num_params, 0.0);
    }

    /**
     * Perform one optimization step - uses const reference to avoid copying
     * @param params Vector of pointers to all model parameters
     * @param num_steps Total number of training steps (for cosine schedule)
     */
    void step(const std::vector<Value*>& params, int num_steps) {
        step_count++;

        // Cosine learning rate decay
        const double lr_t = learning_rate * 0.5 * (1.0 + std::cos(std::numbers::pi * step_count / num_steps));

        for (size_t i = 0; i < params.size(); ++i) {
            Value* p = params[i];

            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * p->grad;

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * p->grad * p->grad;

            // Compute bias-corrected first moment estimate
            const double m_hat = m[i] / (1.0 - std::pow(beta1, step_count));

            // Compute bias-corrected second raw moment estimate
            const double v_hat = v[i] / (1.0 - std::pow(beta2, step_count));

            // Update parameters
            p->data -= lr_t * m_hat / (std::sqrt(v_hat) + eps);

            // Zero gradient
            p->grad = 0.0;
        }
    }

    /**
     * Zero all gradients - uses const reference to avoid copying
     */
    void zero_grad(const std::vector<Value*>& params) {
        for (auto* p : params) {
            p->grad = 0.0;
        }
    }
};

}  // namespace microgpt
