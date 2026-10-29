// qubo_portfolio.cpp - 馬券/銘柄ポートフォリオの QUBO 選択ソルバ
//
// 目的関数(最小化):
//   f(x) = -sum_i score_i x_i + sum_{i<j} P_ij x_i x_j
//   score_i = 正味期待値 - λ·Var_i   (Python側で計算済み)
//   P_ij    = 2λ·Cov_ij              (同上)
// 制約: sum x_i = K を二次ペナルティ A(sum x - K)^2 で埋め込み、
//       K = 1..Kmax を全て解いて真の目的関数が最良の K を採用する。
//
// 使い方: qubo_portfolio.exe input.txt [seed]
// 入力フォーマット(数値のみ, 銘柄名の対応はPython側が持つ):
//   1行目: N Kmax
//   続くN行: score_i
//   続くN行: P の第i行(N個, 対角は無視)
// 出力: 各Kの結果と "BEST K=<k> obj=<f> idx=<i1,i2,...>" を stdout へ
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "qubo.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qubo_portfolio.exe input.txt [seed]\n");
        return 2;
    }
    uint64_t seed = 1;
    if (argc > 2) seed = std::stoull(argv[2]);

    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    int n = 0, kmax = 0;
    in >> n >> kmax;
    if (n <= 0 || kmax <= 0 || kmax > n) {
        std::fprintf(stderr, "invalid header: N=%d Kmax=%d\n", n, kmax);
        return 2;
    }
    std::vector<double> score(n);
    for (auto& s : score) in >> s;
    std::vector<double> pen(static_cast<size_t>(n) * n);
    for (auto& v : pen) in >> v;
    if (!in) {
        std::fprintf(stderr, "input truncated (expected %d scores + %dx%d matrix)\n", n, n, n);
        return 2;
    }

    // 真の目的関数: f(x) = -sum score_i x_i + sum_{i<j} P_ij x_i x_j
    qubo::Problem obj(n);
    for (int i = 0; i < n; ++i) {
        obj.linear(i) = -score[i];
        for (int j = i + 1; j < n; ++j)
            obj.set_pair(i, j, pen[static_cast<size_t>(i) * n + j]);
    }

    qubo::Result best;
    int best_k = -1;
    for (int k = 1; k <= kmax; ++k) {
        qubo::SolveParams sp;
        sp.restarts = 24;
        sp.sweeps = 300;
        sp.seed = seed + static_cast<uint64_t>(k) * 100003;
        qubo::Result r = qubo::solve_exact_k(obj, k, sp);
        if (r.x.empty()) {
            std::printf("K=%d infeasible\n", k);
            continue;
        }
        std::printf("K=%d obj=%.6f\n", k, r.value);
        if (r.value < best.value) {
            best = r;
            best_k = k;
        }
    }

    if (best_k < 0) {
        std::fprintf(stderr, "no feasible solution found\n");
        return 1;
    }
    std::string idx;
    for (int i = 0; i < n; ++i)
        if (best.x[i]) {
            if (!idx.empty()) idx += ',';
            idx += std::to_string(i);
        }
    std::printf("BEST K=%d obj=%.6f idx=%s\n", best_k, best.value, idx.c_str());
    return 0;
}
