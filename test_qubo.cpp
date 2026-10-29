// test_qubo.cpp - QUBOソルバの不変条件テスト
//   1. ランダム密QUBO 20問でアニーリング解 == 総当たり厳密解
//   2. 基数制約(ちょうどK個)のペナルティ埋め込みが制約付き厳密解と一致
//   3. 同一seedで決定的
//   4. 全係数ゼロの縮退問題で値0
#include <cmath>
#include <cstdio>
#include <random>

#include "qubo.hpp"

static int failures = 0;

#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (cond) {                                          \
            std::printf("  ok  : %s\n", msg);                \
        } else {                                             \
            std::printf("  FAIL: %s\n", msg);                \
            ++failures;                                      \
        }                                                    \
    } while (0)

static bool close(double a, double b) {
    return std::abs(a - b) <= 1e-6 * std::max({1.0, std::abs(a), std::abs(b)});
}

int main() {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> coef(-10.0, 10.0);
    std::uniform_int_distribution<int> nsize(8, 18);

    std::printf("[1] random QUBO vs brute force (20 instances)\n");
    int exact = 0;
    for (int t = 0; t < 20; ++t) {
        const int n = nsize(rng);
        qubo::Problem p(n);
        for (int i = 0; i < n; ++i) {
            p.linear(i) = coef(rng);
            for (int j = i + 1; j < n; ++j) p.set_pair(i, j, coef(rng));
        }
        qubo::SolveParams sp;
        sp.seed = 100 + t;
        const auto sol = qubo::solve(p, sp);
        const auto ref = qubo::brute_force(p);
        if (close(sol.value, ref.value)) ++exact;
        else std::printf("       instance %d: anneal=%.6f exact=%.6f\n", t, sol.value, ref.value);
    }
    std::printf("       matched %d/20\n", exact);
    CHECK(exact == 20, "annealing reaches exact optimum on all 20 instances");

    std::printf("[2] cardinality-constrained (K=5, n=14) vs constrained brute force (10 instances)\n");
    {
        const int n = 14, k = 5;
        int feasible = 0, matched = 0;
        for (int t = 0; t < 10; ++t) {
            qubo::Problem p(n);
            for (int i = 0; i < n; ++i) {
                p.linear(i) = -(std::abs(coef(rng)) + 1.0);
                for (int j = i + 1; j < n; ++j) p.set_pair(i, j, coef(rng) * 0.3);
            }
            // 制約付き総当たり
            double ref = 1e300;
            std::vector<uint8_t> x(n);
            for (uint32_t mask = 0; mask < (1u << n); ++mask) {
                if (__builtin_popcount(mask) != k) continue;
                for (int i = 0; i < n; ++i) x[i] = (mask >> i) & 1;
                ref = std::min(ref, p.evaluate(x));
            }
            qubo::SolveParams sp;
            sp.seed = 200 + t;
            const auto sol = qubo::solve_exact_k(p, k, sp);
            int cnt = 0;
            for (auto b : sol.x) cnt += b;
            if (cnt == k) ++feasible;
            if (close(sol.value, ref)) ++matched;
            else std::printf("       instance %d: anneal=%.6f exact=%.6f\n", t, sol.value, ref);
        }
        std::printf("       feasible %d/10, matched %d/10\n", feasible, matched);
        CHECK(feasible == 10, "penalty embedding satisfies the cardinality constraint");
        CHECK(matched == 10, "constrained optimum matches brute force");
    }

    std::printf("[3] determinism\n");
    {
        qubo::Problem p(12);
        for (int i = 0; i < 12; ++i) {
            p.linear(i) = coef(rng);
            for (int j = i + 1; j < 12; ++j) p.set_pair(i, j, coef(rng));
        }
        qubo::SolveParams sp;
        sp.seed = 42;
        const auto a = qubo::solve(p, sp);
        const auto b = qubo::solve(p, sp);
        CHECK(a.value == b.value && a.x == b.x, "same seed gives identical result");
    }

    std::printf("[4] degenerate all-zero problem\n");
    {
        qubo::Problem p(10);
        const auto sol = qubo::solve(p);
        CHECK(sol.value == 0.0, "zero problem solves to zero");
    }

    std::printf("\n%s (failures=%d)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
