// tests/glm_dsa_score_bench.cu - the DSA indexer's pool scores (gf::dsa_score, gf::dsa_score_rows) with several pools
// a warp (STRATA_GLM_DSA_SCORE_PPW) against the original kernel (one pool a warp), bit for bit, and the time of each
// (needs a GPU, no model).  The model's shape: 32 indexer heads of 128, one pool a 4-position block of the context.
//   glm_dsa_score_bench [--parity-only]
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
constexpr int kHeads = 32, kKey = 128, kRowsMax = 4;

}  // namespace

int main(int argc, char** argv) {
    const bool parity_only = argc > 1 && std::string(argv[1]) == "--parity-only";
    std::mt19937 rng(7u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // contexts: a short one, the live median (~64K positions = 16K pools), a long one
    const int pools_of[kRowsMax] = {16000, 23500, 731, 18001};
    std::vector<float*> d_iq(kRowsMax), d_iw(kRowsMax), d_pool(kRowsMax), d_ref(kRowsMax), d_got(kRowsMax);
    for (int r = 0; r < kRowsMax; ++r) {
        std::vector<float> iq(kHeads * kKey), iw(kHeads), pool((size_t) pools_of[r] * kKey);
        for (float& x : iq) x = nd(rng);
        for (float& x : iw) x = nd(rng) * 0.1f;
        for (float& x : pool) x = nd(rng);
        ck(cudaMalloc(&d_iq[r], iq.size() * 4), "malloc");
        ck(cudaMalloc(&d_iw[r], iw.size() * 4), "malloc");
        ck(cudaMalloc(&d_pool[r], pool.size() * 4), "malloc");
        ck(cudaMalloc(&d_ref[r], (size_t) pools_of[r] * 4), "malloc");
        ck(cudaMalloc(&d_got[r], (size_t) pools_of[r] * 4), "malloc");
        ck(cudaMemcpy(d_iq[r], iq.data(), iq.size() * 4, cudaMemcpyHostToDevice), "h2d");
        ck(cudaMemcpy(d_iw[r], iw.data(), iw.size() * 4, cudaMemcpyHostToDevice), "h2d");
        ck(cudaMemcpy(d_pool[r], pool.data(), pool.size() * 4, cudaMemcpyHostToDevice), "h2d");
    }
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto rows_of = [&](int n, std::vector<float*>& out) {
        gf::DsaScoreRows a{};
        for (int r = 0; r < n; ++r) {
            a.iq[r] = d_iq[r];
            a.pooled[r] = d_pool[r];
            a.iw[r] = d_iw[r];
            a.score[r] = out[r];
            a.n_pools[r] = pools_of[r];
        }
        return a;
    };
    auto run = [&](int n, std::vector<float*>& out) {
        if (n == 1)
            gf::dsa_score(d_iq[0], d_pool[0], d_iw[0], kKey, kHeads, pools_of[0], out[0], s);
        else
            gf::dsa_score_rows(rows_of(n, out), n, kKey, kHeads, s);
    };
    auto time_us = [&](int n, std::vector<float*>& out) {
        for (int i = 0; i < 3; ++i) run(n, out);
        ck(cudaEventRecord(e0, s), "record");
        const int reps = 50;
        for (int i = 0; i < reps; ++i) run(n, out);
        ck(cudaEventRecord(e1, s), "record");
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms * 1000.0f / reps;
    };
    int fail = 0, checks = 0;
    for (int n : {1, 2, 4}) {
        gf::dsa_score_config(1);
        run(n, d_ref);
        ck(cudaStreamSynchronize(s), "sync");
        const float t_ref = parity_only ? 0.0f : time_us(n, d_ref);
        for (int ppw : {2, 3, 4, 8, 16, 32}) {
            gf::dsa_score_config(ppw);
            for (int r = 0; r < n; ++r) ck(cudaMemset(d_got[r], 0xff, (size_t) pools_of[r] * 4), "memset");
            run(n, d_got);
            ck(cudaStreamSynchronize(s), "sync");
            bool same = true;
            for (int r = 0; r < n; ++r) {
                std::vector<float> a(pools_of[r]), b(pools_of[r]);
                ck(cudaMemcpy(a.data(), d_ref[r], a.size() * 4, cudaMemcpyDeviceToHost), "d2h");
                ck(cudaMemcpy(b.data(), d_got[r], b.size() * 4, cudaMemcpyDeviceToHost), "d2h");
                same = same && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
            }
            ++checks;
            fail += same ? 0 : 1;
            if (parity_only) {
                std::printf("parity rows=%d ppw=%d %s\n", n, ppw, same ? "PASS" : "FAIL");
                continue;
            }
            const float t = time_us(n, d_got);
            std::printf("bench rows=%d ppw=%-2d %7.1f us | original %7.1f us | %.2fx %s\n", n, ppw, t, t_ref, t_ref / t,
                        same ? "bit-exact" : "DIFFERS");
        }
    }
    gf::dsa_score_config(-1);
    std::printf("glm_dsa_score_bench: %s (%d of %d bitwise equal)\n", fail ? "FAIL" : "PASS", checks - fail, checks);
    return fail ? 1 : 0;
}
