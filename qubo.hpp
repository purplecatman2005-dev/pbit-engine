// qubo.hpp - QUBO solver on top of pbit::Network (quantum-inspired annealing)
//
// QUBO:  minimize  f(x) = sum_i q_i x_i + sum_{i<j} Q_ij x_i x_j,   x in {0,1}^N
// x_i = (1+m_i)/2 で Ising 形式に写像し、pbit のアニーリングで解く。
// pbit のエネルギー E = -1/2 sum_ij J_ij m_i m_j - sum_i h_i m_i に対し
//   J_ij = -Q_ij/4,   h_i = -( q_i/2 + sum_{j!=i} Q_ij/4 )
// とすると E(m) = f(x) + const となり、E 最小化 = f 最小化。
#pragma once
#include <cstdint>
#include <limits>
#include <vector>

#include "pbit.hpp"

namespace qubo {

class Problem {
public:
    explicit Problem(int n) : n_(n), q_(n, 0.0), quad_(static_cast<size_t>(n) * n, 0.0) {}

    int size() const { return n_; }

    double& linear(int i) { return q_[i]; }
    double  linear(int i) const { return q_[i]; }

    // 対称格納。f には i<j のペアとして1回だけ効く
    void set_pair(int i, int j, double v) {
        quad_[static_cast<size_t>(i) * n_ + j] = v;
        quad_[static_cast<size_t>(j) * n_ + i] = v;
    }
    void add_pair(int i, int j, double v) {
        quad_[static_cast<size_t>(i) * n_ + j] += v;
        quad_[static_cast<size_t>(j) * n_ + i] += v;
    }
    double pair(int i, int j) const { return quad_[static_cast<size_t>(i) * n_ + j]; }

    double evaluate(const std::vector<uint8_t>& x) const {
        double f = 0.0;
        for (int i = 0; i < n_; ++i) {
            if (!x[i]) continue;
            f += q_[i];
            const double* row = &quad_[static_cast<size_t>(i) * n_];
            for (int j = i + 1; j < n_; ++j)
                if (x[j]) f += row[j];
        }
        return f;
    }

private:
    int n_;
    std::vector<double> q_;
    std::vector<double> quad_;  // row-major n x n, symmetric, diagonal unused
};

struct Result {
    std::vector<uint8_t> x;
    double value = std::numeric_limits<double>::infinity();
};

struct SolveParams {
    int restarts = 32;
    int sweeps = 400;
    double beta_start = 0.1;   // 正規化後の係数スケールに対する値
    double beta_end = 15.0;
    uint64_t seed = 1;
};

// 単一ビット反転の山下り。アニーリング解の仕上げ(局所最適まで落とす)
inline void polish(const Problem& p, std::vector<uint8_t>& x) {
    const int n = p.size();
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n; ++i) {
            // x_i を反転したときの f の変化量
            double delta = p.linear(i);
            for (int j = 0; j < n; ++j)
                if (j != i && x[j]) delta += p.pair(i, j);
            if (x[i]) delta = -delta;   // 1->0 なら符号反転
            if (delta < -1e-12) {
                x[i] ^= 1;
                improved = true;
            }
        }
    }
}

inline Result solve(const Problem& p, const SolveParams& sp = {}) {
    const int n = p.size();
    pbit::Network net(n, sp.seed);

    // Ising 写像 + 係数正規化(betaスケールを問題非依存にする)
    double scale = 0.0;
    std::vector<double> h(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            if (j != i) s += p.pair(i, j);
        h[i] = -(p.linear(i) / 2.0 + s / 4.0);
        scale = std::max(scale, std::abs(h[i]));
    }
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            scale = std::max(scale, std::abs(p.pair(i, j)) / 4.0);
    if (scale <= 0.0) scale = 1.0;

    for (int i = 0; i < n; ++i) {
        net.h(i) = h[i] / scale;
        for (int j = i + 1; j < n; ++j) {
            const double jij = -p.pair(i, j) / 4.0 / scale;
            net.J(i, j) = jij;
            net.J(j, i) = jij;
        }
    }

    Result best;
    std::vector<uint8_t> x(n);
    for (int r = 0; r < sp.restarts; ++r) {
        net.randomize_state();
        net.anneal(sp.beta_start, sp.beta_end, sp.sweeps);
        const auto& m = net.state();
        for (int i = 0; i < n; ++i) x[i] = (m[i] > 0) ? 1 : 0;
        polish(p, x);
        const double f = p.evaluate(x);
        if (f < best.value) {
            best.value = f;
            best.x = x;
        }
    }
    return best;
}

// 1-in/1-out スワップの山下り。選択数を保ちながら真の目的関数を下げる。
// 基数制約のペナルティ埋め込みでは、スワップは制約違反状態を経由するため
// アニーリングだけでは越えにくい。この仕上げが実行可能解間の移動を担う。
inline void polish_swap(const Problem& p, std::vector<uint8_t>& x) {
    const int n = p.size();
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n; ++i) {
            if (!x[i]) continue;
            for (int j = 0; j < n; ++j) {
                if (x[j]) continue;
                double delta = p.linear(j) - p.linear(i);
                for (int k = 0; k < n; ++k)
                    if (x[k] && k != i && k != j)
                        delta += p.pair(j, k) - p.pair(i, k);
                if (delta < -1e-12) {
                    x[i] = 0;
                    x[j] = 1;
                    improved = true;
                    break;   // i はもう選択されていない
                }
            }
        }
    }
}

// 選択数を貪欲にKへ修復する(過剰なら寄与最悪を外し、不足なら追加コスト最小を入れる)
inline void repair_to_k(const Problem& p, std::vector<uint8_t>& x, int k) {
    const int n = p.size();
    int cnt = 0;
    for (auto b : x) cnt += b;
    auto marginal = [&](int i) {
        double c = p.linear(i);
        for (int j = 0; j < n; ++j)
            if (j != i && x[j]) c += p.pair(i, j);
        return c;
    };
    while (cnt > k) {
        int worst = -1;
        double wv = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i)
            if (x[i]) {
                const double c = marginal(i);
                if (c > wv) { wv = c; worst = i; }
            }
        x[worst] = 0;
        --cnt;
    }
    while (cnt < k) {
        int bestj = -1;
        double bv = std::numeric_limits<double>::infinity();
        for (int j = 0; j < n; ++j)
            if (!x[j]) {
                const double c = marginal(j);
                if (c < bv) { bv = c; bestj = j; }
            }
        x[bestj] = 1;
        ++cnt;
    }
}

// 「ちょうどK個選ぶ」制約付き最小化。
// ペナルティ A(sum x - K)^2 を埋め込み、リスタートごとに
// 貪欲修復 → スワップ降下で実行可能な局所最適を作り、最良を返す。
inline Result solve_exact_k(const Problem& p, int k, SolveParams sp = {}) {
    const int n = p.size();
    double max_gain = 0.0;
    for (int i = 0; i < n; ++i) {
        double row = std::abs(p.linear(i));
        for (int j = 0; j < n; ++j)
            if (j != i) row += std::abs(p.pair(i, j));
        max_gain = std::max(max_gain, row);
    }
    const double a = 2.0 * max_gain + 1.0;

    Problem pen(n);
    for (int i = 0; i < n; ++i) {
        pen.linear(i) = p.linear(i) + a * (1.0 - 2.0 * k);
        for (int j = i + 1; j < n; ++j)
            pen.set_pair(i, j, p.pair(i, j) + 2.0 * a);
    }

    Result best;
    const int outer = std::max(1, sp.restarts / 4);
    for (int r = 0; r < outer; ++r) {
        SolveParams s = sp;
        s.restarts = 4;
        s.seed = sp.seed + static_cast<uint64_t>(r) * 104729 + 1;
        Result cand = solve(pen, s);
        repair_to_k(p, cand.x, k);
        polish_swap(p, cand.x);
        const double f = p.evaluate(cand.x);
        if (f < best.value) {
            best.value = f;
            best.x = cand.x;
        }
    }
    return best;
}

// 総当たり厳密解(テスト用, n <= 26)
inline Result brute_force(const Problem& p) {
    const int n = p.size();
    Result best;
    std::vector<uint8_t> x(n);
    for (uint64_t mask = 0; mask < (uint64_t{1} << n); ++mask) {
        for (int i = 0; i < n; ++i) x[i] = (mask >> i) & 1;
        const double f = p.evaluate(x);
        if (f < best.value) {
            best.value = f;
            best.x = x;
        }
    }
    return best;
}

}  // namespace qubo
