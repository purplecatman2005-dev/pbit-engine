// patterns.hpp - built-in digit bitmaps, noise, pseudo-inverse learning, classification
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pbit.hpp"

namespace pbit {

constexpr int kW = 10;
constexpr int kH = 12;
constexpr int kN = kW * kH;
constexpr int kDigits = 10;

// 10x12 block-font digits, '#' = +1, '.' = -1
inline constexpr std::array<std::array<std::string_view, kH>, kDigits> kDigitRows{{
    {"..######..", ".##....##.", ".##....##.", ".##....##.", ".##....##.", ".##....##.",
     ".##....##.", ".##....##.", ".##....##.", ".##....##.", ".##....##.", "..######.."},
    {"....##....", "...###....", "..####....", "....##....", "....##....", "....##....",
     "....##....", "....##....", "....##....", "....##....", "....##....", "..######.."},
    {"..######..", ".##....##.", ".......##.", ".......##.", "......##..", ".....##...",
     "....##....", "...##.....", "..##......", ".##.......", ".##.......", ".########."},
    {"..######..", ".##....##.", ".......##.", ".......##.", "...####...", "...####...",
     ".......##.", ".......##.", ".......##.", ".......##.", ".##....##.", "..######.."},
    {".....###..", "....####..", "...##.##..", "..##..##..", ".##...##..", ".##...##..",
     ".########.", "......##..", "......##..", "......##..", "......##..", "......##.."},
    {".########.", ".##.......", ".##.......", ".##.......", ".#######..", ".......##.",
     ".......##.", ".......##.", ".......##.", ".......##.", ".##....##.", "..######.."},
    {"..######..", ".##....##.", ".##.......", ".##.......", ".#######..", ".##....##.",
     ".##....##.", ".##....##.", ".##....##.", ".##....##.", ".##....##.", "..######.."},
    {".########.", ".......##.", ".......##.", "......##..", "......##..", ".....##...",
     ".....##...", "....##....", "....##....", "...##.....", "...##.....", "...##....."},
    {"..######..", ".##....##.", ".##....##.", ".##....##.", "..######..", "..######..",
     ".##....##.", ".##....##.", ".##....##.", ".##....##.", ".##....##.", "..######.."},
    {"..######..", ".##....##.", ".##....##.", ".##....##.", ".##....##.", "..#######.",
     ".......##.", ".......##.", ".......##.", ".......##.", ".##....##.", "..######.."},
}};

using Pattern = std::vector<int8_t>;  // kN values in {-1,+1}

inline std::vector<Pattern> digit_patterns() {
    std::vector<Pattern> ps(kDigits, Pattern(kN));
    for (int d = 0; d < kDigits; ++d) {
        for (int r = 0; r < kH; ++r) {
            if (kDigitRows[d][r].size() != kW)
                throw std::runtime_error("bad bitmap row: digit " + std::to_string(d));
            for (int c = 0; c < kW; ++c)
                ps[d][r * kW + c] = (kDigitRows[d][r][c] == '#') ? int8_t{1} : int8_t{-1};
        }
    }
    return ps;
}

// flip each pixel independently with probability `rate`
inline Pattern add_flip_noise(const Pattern& p, double rate, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Pattern out = p;
    for (auto& v : out)
        if (u(rng) < rate) v = static_cast<int8_t>(-v);
    return out;
}

// Gauss-Jordan inverse of a small dense matrix (p x p, row-major)
inline std::vector<double> invert(std::vector<double> a, int p) {
    std::vector<double> inv(static_cast<size_t>(p) * p, 0.0);
    for (int i = 0; i < p; ++i) inv[static_cast<size_t>(i) * p + i] = 1.0;
    for (int col = 0; col < p; ++col) {
        int piv = col;
        for (int r = col + 1; r < p; ++r)
            if (std::abs(a[static_cast<size_t>(r) * p + col]) >
                std::abs(a[static_cast<size_t>(piv) * p + col])) piv = r;
        if (std::abs(a[static_cast<size_t>(piv) * p + col]) < 1e-12)
            throw std::runtime_error("singular overlap matrix");
        for (int c = 0; c < p; ++c) {
            std::swap(a[static_cast<size_t>(piv) * p + c], a[static_cast<size_t>(col) * p + c]);
            std::swap(inv[static_cast<size_t>(piv) * p + c], inv[static_cast<size_t>(col) * p + c]);
        }
        const double d = a[static_cast<size_t>(col) * p + col];
        for (int c = 0; c < p; ++c) {
            a[static_cast<size_t>(col) * p + c] /= d;
            inv[static_cast<size_t>(col) * p + c] /= d;
        }
        for (int r = 0; r < p; ++r) {
            if (r == col) continue;
            const double f = a[static_cast<size_t>(r) * p + col];
            if (f == 0.0) continue;
            for (int c = 0; c < p; ++c) {
                a[static_cast<size_t>(r) * p + c] -= f * a[static_cast<size_t>(col) * p + c];
                inv[static_cast<size_t>(r) * p + c] -= f * inv[static_cast<size_t>(col) * p + c];
            }
        }
    }
    return inv;
}

// pseudo-inverse rule: J = (1/N) * Xi * Q^-1 * Xi^T,  Q_uv = (1/N) xi_u . xi_v
// makes every stored pattern an exact fixed point of the beta->inf dynamics
inline void train_pseudo_inverse(Network& net, const std::vector<Pattern>& pats) {
    const int p = static_cast<int>(pats.size());
    const int n = net.size();
    std::vector<double> q(static_cast<size_t>(p) * p, 0.0);
    for (int u = 0; u < p; ++u)
        for (int v = 0; v < p; ++v) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += pats[u][i] * pats[v][i];
            q[static_cast<size_t>(u) * p + v] = s / n;
        }
    const std::vector<double> qi = invert(std::move(q), p);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) { net.J(i, j) = 0.0; continue; }
            double s = 0.0;
            for (int u = 0; u < p; ++u)
                for (int v = 0; v < p; ++v)
                    s += pats[u][i] * qi[static_cast<size_t>(u) * p + v] * pats[v][j];
            net.J(i, j) = s / n;
        }
    }
}

// recall by annealed Gibbs sampling of the posterior:
//   prior   = stored patterns via coupling matrix J
//   evidence = noisy image injected as bias field h_i = lambda * noisy_i
// the bias keeps the observed image from being erased by thermal noise,
// while the p-bit fluctuations let the state escape spurious mixtures
inline Pattern recognize(Network& net, const Pattern& noisy, double lambda = 0.6,
                         double beta_start = 1.0, double beta_end = 6.0,
                         int sweeps = 40, int freeze_sweeps = 10) {
    const int n = net.size();
    for (int i = 0; i < n; ++i) net.h(i) = lambda * noisy[i];
    net.set_state(noisy);
    net.anneal(beta_start, beta_end, sweeps);
    // freeze: decay the evidence to zero while cooling further, so the state
    // relaxes into a clean stored pattern without leaving the correct basin
    const int decay_span = std::max(1, freeze_sweeps - 1);
    for (int s = 0; s < freeze_sweeps; ++s) {
        const double f = lambda * (freeze_sweeps - 1 - s) / decay_span;
        for (int i = 0; i < n; ++i) net.h(i) = f * noisy[i];
        net.sweep(beta_end + s);
    }
    for (int i = 0; i < n; ++i) net.h(i) = 0.0;
    return net.state();
}

// classify by maximum overlap  m . xi_u / N  (returns digit index)
inline int classify(const Pattern& m, const std::vector<Pattern>& pats, double* best_overlap = nullptr) {
    int arg = 0;
    double best = -2.0;
    for (int u = 0; u < static_cast<int>(pats.size()); ++u) {
        double s = 0.0;
        for (size_t i = 0; i < m.size(); ++i) s += m[i] * pats[u][i];
        s /= static_cast<double>(m.size());
        if (s > best) { best = s; arg = u; }
    }
    if (best_overlap) *best_overlap = best;
    return arg;
}

inline std::string render(const Pattern& m) {
    std::string out;
    out.reserve((kW + 1) * kH);
    for (int r = 0; r < kH; ++r) {
        for (int c = 0; c < kW; ++c) out += (m[r * kW + c] > 0) ? '#' : '.';
        out += '\n';
    }
    return out;
}

}  // namespace pbit
