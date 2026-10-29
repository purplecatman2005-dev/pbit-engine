// test_pbit.cpp - invariant tests for the p-bit engine
#include <cmath>
#include <cstdio>
#include <vector>

#include "patterns.hpp"

using namespace pbit;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// 1. single free p-bit: empirical mean must match tanh(beta*h) within 4 sigma
void test_pbit_statistics() {
    const double beta = 1.0, bias = 0.7;
    Network net(1, 111);
    net.h(0) = bias;
    const int samples = 200000;
    long sum = 0;
    for (int s = 0; s < samples; ++s) {
        net.sweep(beta);
        sum += net.state()[0];
    }
    const double mean = static_cast<double>(sum) / samples;
    const double expect = std::tanh(beta * bias);
    const double sigma = std::sqrt((1.0 - expect * expect) / samples);
    const double dev = std::abs(mean - expect);
    std::printf("  mean=%.4f expect=%.4f (%.1f sigma)\n", mean, expect, dev / sigma);
    check(dev < 4.0 * sigma, "free p-bit mean matches tanh(beta*h)");
}

// 2. at high beta the dynamics are quasi-deterministic: energy never increases
void test_energy_descent() {
    const auto pats = digit_patterns();
    Network net(kN, 222);
    train_pseudo_inverse(net, pats);
    net.randomize_state();
    double prev = net.energy();
    bool monotone = true;
    for (int s = 0; s < 30; ++s) {
        net.sweep(100.0);
        const double e = net.energy();
        if (e > prev + 1e-9) monotone = false;
        prev = e;
    }
    check(monotone, "energy non-increasing at beta=100");
}

// 3. every stored digit is a fixed point at high beta
void test_pattern_stability() {
    const auto pats = digit_patterns();
    Network net(kN, 333);
    train_pseudo_inverse(net, pats);
    bool all_stable = true;
    for (int d = 0; d < kDigits; ++d) {
        net.set_state(pats[d]);
        for (int s = 0; s < 10; ++s) net.sweep(100.0);
        double ov = 0.0;
        classify(net.state(), pats, &ov);
        if (classify(net.state(), pats) != d || ov < 0.9999) all_stable = false;
    }
    check(all_stable, "all 10 stored digits are stable fixed points");
}

// 4. end-to-end recall: 20% noise, fixed seed, expect >= 90% correct
void test_noisy_recall() {
    const auto pats = digit_patterns();
    Network net(kN, 444);
    train_pseudo_inverse(net, pats);
    std::mt19937_64 noise_rng(4444);
    int correct = 0, total = 0;
    for (int d = 0; d < kDigits; ++d) {
        for (int t = 0; t < 10; ++t) {
            const Pattern noisy = add_flip_noise(pats[d], 0.20, noise_rng);
            if (classify(recognize(net, noisy), pats) == d) ++correct;
            ++total;
        }
    }
    std::printf("  recall %d/%d at 20%% noise\n", correct, total);
    check(correct >= total * 9 / 10, "recall accuracy >= 90% at 20% noise");
}

}  // namespace

int main() {
    test_pbit_statistics();
    test_energy_descent();
    test_pattern_stability();
    test_noisy_recall();
    std::printf(g_failures ? "\n%d test(s) FAILED\n" : "\nall tests passed\n", g_failures);
    return g_failures ? 1 : 0;
}
