// pbit.hpp - probabilistic bit (p-bit) network core
// p-bit update rule (Camsari et al.):  m_i = sgn( tanh(beta * I_i) - r ),  r ~ U(-1,1)
// where I_i = sum_j J_ij m_j + h_i  is the local field.
// beta -> 0   : pure random bits (quantum-like superposition analogue)
// beta -> inf : deterministic Hopfield dynamics (measurement / collapse analogue)
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace pbit {

class Network {
public:
    Network(int n, uint64_t seed)
        : n_(n), j_(static_cast<size_t>(n) * n, 0.0), h_(n, 0.0),
          m_(n, 1), rng_(seed), order_(n) {
        for (int i = 0; i < n_; ++i) order_[i] = i;
    }

    int size() const { return n_; }

    double& J(int i, int j) { return j_[static_cast<size_t>(i) * n_ + j]; }
    double  J(int i, int j) const { return j_[static_cast<size_t>(i) * n_ + j]; }
    double& h(int i) { return h_[i]; }

    void set_state(const std::vector<int8_t>& m) { m_ = m; }
    const std::vector<int8_t>& state() const { return m_; }

    double local_field(int i) const {
        const double* row = &j_[static_cast<size_t>(i) * n_];
        double I = h_[i];
        for (int j = 0; j < n_; ++j) I += row[j] * m_[j];
        return I;
    }

    // E = -1/2 sum_ij J_ij m_i m_j - sum_i h_i m_i
    double energy() const {
        double e = 0.0;
        for (int i = 0; i < n_; ++i) {
            const double* row = &j_[static_cast<size_t>(i) * n_];
            double s = 0.0;
            for (int j = 0; j < n_; ++j) s += row[j] * m_[j];
            e -= 0.5 * s * m_[i] + h_[i] * m_[i];
        }
        return e;
    }

    // one asynchronous sweep: every p-bit updated once in random order
    void sweep(double beta) {
        std::shuffle(order_.begin(), order_.end(), rng_);
        for (int i : order_) {
            const double I = local_field(i);
            m_[i] = (std::tanh(beta * I) > unif_(rng_)) ? int8_t{1} : int8_t{-1};
        }
    }

    // geometric annealing schedule beta_start -> beta_end over `sweeps` sweeps
    void anneal(double beta_start, double beta_end, int sweeps) {
        const double ratio = std::pow(beta_end / beta_start,
                                      1.0 / std::max(1, sweeps - 1));
        double beta = beta_start;
        for (int s = 0; s < sweeps; ++s) {
            sweep(beta);
            beta *= ratio;
        }
    }

    void randomize_state() {
        for (auto& m : m_) m = (unif_(rng_) > 0.0) ? int8_t{1} : int8_t{-1};
    }

    std::mt19937_64& rng() { return rng_; }

private:
    int n_;
    std::vector<double> j_;   // row-major n x n, symmetric, zero diagonal
    std::vector<double> h_;
    std::vector<int8_t> m_;   // states in {-1,+1}
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> unif_{-1.0, 1.0};
    std::vector<int> order_;
};

}  // namespace pbit
