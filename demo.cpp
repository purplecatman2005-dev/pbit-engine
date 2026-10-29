// demo.cpp - p-bit image recognition demo
// train 10 digits -> corrupt with pixel-flip noise -> p-bit annealing -> classify
#include <cstdio>
#include <vector>

#include "patterns.hpp"

using namespace pbit;

namespace {

void show_side_by_side(const Pattern& a, const Pattern& b, const Pattern& c) {
    std::printf("%-*s   %-*s   %-*s\n", kW, "original", kW, "noisy", kW, "recovered");
    for (int r = 0; r < kH; ++r) {
        for (int c2 = 0; c2 < kW; ++c2) std::putchar(a[r * kW + c2] > 0 ? '#' : '.');
        std::printf("   ");
        for (int c2 = 0; c2 < kW; ++c2) std::putchar(b[r * kW + c2] > 0 ? '#' : '.');
        std::printf("   ");
        for (int c2 = 0; c2 < kW; ++c2) std::putchar(c[r * kW + c2] > 0 ? '#' : '.');
        std::putchar('\n');
    }
}

}  // namespace

int main() {
    const auto pats = digit_patterns();
    Network net(kN, /*seed=*/20260712);
    train_pseudo_inverse(net, pats);
    std::printf("p-bit network: N=%d p-bits, %d stored digits, pseudo-inverse rule\n",
                kN, kDigits);
    std::printf("recall: evidence bias lambda=0.6, anneal beta 1.0 -> 6.0 (40 sweeps),\n"
                "        then 10 freeze sweeps with decaying evidence\n\n");

    // --- single example, visualized ---
    {
        std::mt19937_64 noise_rng(7);
        const Pattern noisy = add_flip_noise(pats[8], 0.25, noise_rng);
        const Pattern recovered = recognize(net, noisy);
        double ov = 0.0;
        const int cls = classify(recovered, pats, &ov);
        std::printf("example: digit 8 with 25%% pixel-flip noise\n");
        show_side_by_side(pats[8], noisy, recovered);
        std::printf("-> classified as %d (overlap %.3f)\n\n", cls, ov);
    }

    // --- accuracy sweep over noise levels ---
    constexpr int kTrials = 100;  // per digit per noise level
    const double noise_levels[] = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40};
    std::printf("accuracy (%d trials x %d digits per level):\n", kTrials, kDigits);
    std::printf("  noise   accuracy   mean overlap\n");
    std::mt19937_64 noise_rng(42);
    for (const double rate : noise_levels) {
        int correct = 0;
        double ov_sum = 0.0;
        for (int d = 0; d < kDigits; ++d) {
            for (int t = 0; t < kTrials; ++t) {
                const Pattern noisy = add_flip_noise(pats[d], rate, noise_rng);
                const Pattern rec = recognize(net, noisy);
                double ov = 0.0;
                if (classify(rec, pats, &ov) == d) ++correct;
                ov_sum += ov;
            }
        }
        const int total = kTrials * kDigits;
        std::printf("  %4.0f%%   %5.1f%%      %.3f\n", rate * 100,
                    100.0 * correct / total, ov_sum / total);
    }
    return 0;
}
