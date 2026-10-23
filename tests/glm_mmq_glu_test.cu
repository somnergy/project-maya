// tests/glm_mmq_glu_test.cu - the prompt path's fused gate/up + SwiGLU product (mmq::Context::run_glu) against the
// plain gate/up product followed by the separate SwiGLU kernel (glmb::swiglu_rows): every expert type the GLM prompt
// path multiplies with MMQ, the SwiGLU clamped and unclamped, bit for bit where the plain product sums in the same order
// (see main).
// Synthetic weights from ggml's own quantizers; no model needed.
//
//   glm_mmq_glu_test           the parity checks (ctest; skipped where mmq::glu_supported is false)
//   glm_mmq_glu_test --bench   Maya-L's shape (IQ3_S, n_embd 4096, n_ff 2048, 288 experts, 8192 tokens x 8): the time
//                              of the fused product against the plain product + the SwiGLU kernel
//   glm_mmq_glu_test --tiles   Maya-L's gate/up + down at a prompt chunk's routing (512-8192 tokens): every MMQ column
//                              tile (Product::opt_rows) and mmq::tile_cols' picks, timed and checked bit for bit
#include "strata/kernels/glm_batch.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace gb = strata::kernels::glmb;

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_mmq_glu_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* dev(size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return (T*) p;
}

// n_exp experts' [2 n_ff, n_embd] gate-then-up matrices, weights ~ N(0, 0.05) through ggml's quantizer (an importance
// matrix of ones for the types that need one), one after another
std::vector<uint8_t> make_experts(int type, int n_exp, int n_embd, int n_ff, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.05f);
    const size_t eb = mmq::matrix_bytes(type, 2 * n_ff, n_embd);
    std::vector<uint8_t> host(eb * (size_t) n_exp);
    std::vector<float> f((size_t) 2 * n_ff * n_embd);
    const std::vector<float> imatrix((size_t) n_embd, 1.0f);
    for (int e = 0; e < n_exp; ++e) {
        for (auto& v : f) v = nd(rng);
        const size_t wrote = ggml_quantize_chunk((ggml_type) type, f.data(), host.data() + eb * (size_t) e, 0, 2 * n_ff,
                                                 n_embd,
                                                 ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
        if (wrote != eb) {
            std::fprintf(stderr, "glm_mmq_glu_test: %s: ggml wrote %zu bytes, MMQ reads %zu\n",
                         ggml_type_name((ggml_type) type), wrote, eb);
            std::exit(1);
        }
    }
    return host;
}

struct Case {
    const void* w = nullptr;   // device experts
    size_t eb = 0;
    int type = 0, n_exp = 0, n_embd = 0, n_ff = 0, total = 0, max_rows = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
};

mmq::Product product(const Case& c, float* dst) {
    mmq::Product p;
    p.w = c.w;
    p.type = c.type;
    p.w_rows = 2 * c.n_ff;
    p.w_cols = c.n_embd;
    p.expert_bytes = c.eb;
    p.n = c.n_exp;
    p.xq = c.xq;
    p.bounds = c.bounds;
    p.ids = c.ids;
    p.total_rows = c.total;
    p.max_rows = c.max_rows;
    p.dst = dst;
    p.ld_dst = 2 * c.n_ff;
    return p;
}

int bench(cudaStream_t s) {
    const int type = GGML_TYPE_IQ3_S, n_exp = 288, n_embd = 4096, n_ff = 2048, T = 8192 * 8;
    if (!mmq::glu_supported(type, n_ff)) {
        std::printf("glm_mmq_glu_test: no fused product on this device/build\n");
        return 0;
    }
    std::mt19937 rng(7);
    // random codes (the speed does not depend on the values); every expert ~ T / n_exp rows
    const size_t eb = mmq::matrix_bytes(type, 2 * n_ff, n_embd);
    std::vector<uint8_t> hw(eb * n_exp);
    for (auto& b : hw) b = (uint8_t) rng();
    for (size_t o = 0; o < hw.size(); o += 110) hw[o + 1] = 0x10;   // IQ3_S d: small, finite
    uint8_t* w = dev<uint8_t>(hw.size());
    check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "upload");
    std::vector<int32_t> hb(n_exp + 1);
    for (int e = 0; e <= n_exp; ++e) hb[e] = (int) ((int64_t) T * e / n_exp);
    int max_rows = 0;
    for (int e = 0; e < n_exp; ++e) max_rows = std::max(max_rows, hb[e + 1] - hb[e]);
    int32_t* bounds = dev<int32_t>(hb.size());
    check(cudaMemcpy(bounds, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice), "bounds");
    int32_t* ids = dev<int32_t>(T);
    mmq::iota(ids, T, s);
    float* x = dev<float>((size_t) T * n_embd);
    check(cudaMemset(x, 0, (size_t) T * n_embd * 4), "x");
    uint8_t* xq = dev<uint8_t>(mmq::q8_bytes(T, n_embd));
    mmq::quantize(x, nullptr, xq, type, n_embd, n_embd, T, s);
    float* gu = dev<float>((size_t) T * 2 * n_ff);
    Case c{w, eb, type, n_exp, n_embd, n_ff, T, max_rows, xq, bounds, ids};
    mmq::Context ctx;
    const mmq::Product p = product(c, gu);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    // 0: the plain product, 1: the SwiGLU kernel, 2: both (today's path), 3: the fused product
    const char* what[] = {"gate/up product", "SwiGLU kernel", "product + SwiGLU kernel", "fused product"};
    float best[4] = {1e9f, 1e9f, 1e9f, 1e9f};
    for (int rep = 0; rep < 8; ++rep)
        for (int k = 0; k < 4; ++k) {
            cudaEventRecord(e0, s);
            if (k == 0 || k == 2) ctx.run(p, s);
            if (k == 1 || k == 2) gb::swiglu_rows(gu, gu, T, n_ff, INFINITY, s, 2 * n_ff);
            if (k == 3) ctx.run_glu(p, INFINITY, s);
            cudaEventRecord(e1, s);
            check(cudaEventSynchronize(e1), "bench");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (rep > 0) best[k] = std::min(best[k], ms);
        }
    for (int k = 0; k < 4; ++k)
        std::printf("glm_mmq_glu_test --bench: IQ3_S %d x %d, %d experts, %d rows: %-24s %6.2f ms\n", 2 * n_ff, n_embd,
                    n_exp, T, what[k], best[k]);
    return 0;
}
// --tiles: Maya-L's expert products (gate/up IQ3_S fused with the SwiGLU, down IQ4_XS; n_embd 4096, n_ff 2048) over
// 256 experts with a prompt chunk's routing - T tokens x 8 routes over experts of uneven popularity (gamma weights,
// CV 0.8) - at every MMQ column tile J (Product::opt_rows) and at mmq::tile_cols' pick for a few fixed costs, against
// the tile for the busiest expert (opt_rows 0, the default): the time of each and whether every output is bit for bit
// the default's
int tiles(cudaStream_t s) {
    const int gu_type = GGML_TYPE_IQ3_S, d_type = GGML_TYPE_IQ4_XS, n_exp = 256, n_embd = 4096, n_ff = 2048, K = 8;
    if (!mmq::glu_supported(gu_type, n_ff)) {
        std::printf("glm_mmq_glu_test --tiles: no fused product on this device/build\n");
        return 0;
    }
    std::mt19937 rng(23);
    // random codes with a small, finite scale in each block (the speed does not depend on the values)
    const auto weights = [&](int type, int64_t rows, int64_t cols, size_t block) {
        const size_t eb = mmq::matrix_bytes(type, rows, cols);
        std::vector<uint8_t> hw(eb * n_exp);
        for (auto& b : hw) b = (uint8_t) rng();
        for (size_t o = 0; o + 1 < hw.size(); o += block) hw[o + 1] = 0x10;
        uint8_t* w = dev<uint8_t>(hw.size());
        check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "upload");
        return std::make_pair(w, eb);
    };
    const auto [wgu, ebgu] = weights(gu_type, 2 * n_ff, n_embd, 110);   // block_iq3_s: 110 bytes, d first
    const auto [wd, ebd] = weights(d_type, n_embd, n_ff, 136);           // block_iq4_xs: 136 bytes, d first
    std::gamma_distribution<double> gd(1.0 / (0.8 * 0.8), 1.0);
    std::vector<double> pop(n_exp);
    double sum = 0.0;
    for (auto& p : pop) sum += (p = gd(rng));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    mmq::Context ctx;
    std::printf("glm_mmq_glu_test --tiles: IQ3_S gate/up (fused) + IQ4_XS down, %d x %d / %d, %d experts, 8 routes a "
                "token; ms are the best of 5; 'differ' counts outputs not bit for bit the default's\n",
                2 * n_ff, n_embd, n_ff, n_exp);
    for (const int T : {512, 1024, 2048, 4096, 8192}) {
        // rows per expert: T * K routes spread by popularity (largest remainders), the order as in a pool partition
        std::vector<int32_t> rows(n_exp);
        int got = 0;
        for (int e = 0; e < n_exp; ++e) got += rows[e] = (int32_t) std::floor(pop[e] / sum * T * K);
        for (int e = 0; got < T * K; e = (e + 1) % n_exp, ++got) ++rows[e];
        std::vector<int32_t> hb(n_exp + 1, 0);
        for (int e = 0; e < n_exp; ++e) hb[e + 1] = hb[e] + rows[e];
        const int R = hb[n_exp], max_rows = *std::max_element(rows.begin(), rows.end());
        int nz = 0;
        for (int r : rows) nz += r > 0;
        int32_t* bounds = dev<int32_t>(hb.size());
        check(cudaMemcpy(bounds, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice), "bounds");
        int32_t* ids = dev<int32_t>(R);
        mmq::iota(ids, R, s);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> hx((size_t) R * n_embd);
        for (auto& v : hx) v = nd(rng);
        float* x = dev<float>(hx.size());
        check(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        uint8_t* xq = dev<uint8_t>(mmq::q8_bytes(R, n_embd));
        mmq::quantize(x, nullptr, xq, gu_type, n_embd, n_embd, R, s);
        float* gu = dev<float>((size_t) R * 2 * n_ff);
        float* out = dev<float>((size_t) R * n_embd);
        uint8_t* hq = dev<uint8_t>(mmq::q8_bytes(R, n_ff));
        std::vector<float> ref_h, ref_o, hh((size_t) R * 2 * n_ff), ho((size_t) R * n_embd);
        std::printf("  T %5d: %d rows, %d experts with rows (mean %.0f, max %d)\n", T, R, nz, (double) R / nz, max_rows);
        // modes: 0 = the default (opt_rows 0), > 0 = that J, < 0 = tile_cols with fixed cost -mode
        std::vector<int> modes = {0, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, -8, -16, -32, -64};
        double t_def = 0.0;
        for (const int mode : modes) {
            const int opt = mode > 0 ? std::min(mode, max_rows) : mode < 0 ? mmq::tile_cols(hb.data(), n_exp, max_rows, -mode) : 0;
            mmq::Product g;
            g.w = wgu; g.type = gu_type; g.w_rows = 2 * n_ff; g.w_cols = n_embd; g.expert_bytes = ebgu; g.n = n_exp;
            g.xq = xq; g.bounds = bounds; g.ids = ids; g.total_rows = R; g.max_rows = max_rows; g.dst = gu;
            g.ld_dst = 2 * n_ff; g.opt_rows = opt;
            mmq::Product d;
            d.w = wd; d.type = d_type; d.w_rows = n_embd; d.w_cols = n_ff; d.expert_bytes = ebd; d.n = n_exp;
            d.xq = hq; d.bounds = bounds; d.ids = ids; d.total_rows = R; d.max_rows = max_rows; d.dst = out;
            d.ld_dst = n_embd; d.opt_rows = opt;
            float best_g = 1e9f, best_d = 1e9f;
            for (int rep = 0; rep < 6; ++rep) {
                float ms = 0.f;
                cudaEventRecord(e0, s);
                ctx.run_glu(g, INFINITY, s);
                cudaEventRecord(e1, s);
                check(cudaEventSynchronize(e1), "gate/up");
                cudaEventElapsedTime(&ms, e0, e1);
                if (rep > 0) best_g = std::min(best_g, ms);
                if (rep == 0) mmq::quantize(gu, nullptr, hq, d_type, n_ff, 2 * n_ff, R, s);   // (h: the first n_ff)
                cudaEventRecord(e0, s);
                ctx.run(d, s);
                cudaEventRecord(e1, s);
                check(cudaEventSynchronize(e1), "down");
                cudaEventElapsedTime(&ms, e0, e1);
                if (rep > 0) best_d = std::min(best_d, ms);
            }
            check(cudaMemcpy(hh.data(), gu, hh.size() * 4, cudaMemcpyDeviceToHost), "h");
            check(cudaMemcpy(ho.data(), out, ho.size() * 4, cudaMemcpyDeviceToHost), "out");
            int dh = 0, dout = 0, finite = 1;
            if (mode == 0) {
                ref_h = hh;
                ref_o = ho;
                t_def = best_g + best_d;
            } else {
                for (int r = 0; r < R; ++r)
                    for (int k = 0; k < n_ff; ++k)
                        dh += std::memcmp(&hh[(size_t) r * 2 * n_ff + k], &ref_h[(size_t) r * 2 * n_ff + k], 4) != 0;
                for (size_t i = 0; i < ho.size(); ++i) dout += std::memcmp(&ho[i], &ref_o[i], 4) != 0;
            }
            for (size_t i = 0; i < ho.size() && finite; i += 997) finite = std::isfinite(ho[i]);
            char name[48];
            if (mode == 0) std::snprintf(name, sizeof name, "default");
            else if (mode > 0) std::snprintf(name, sizeof name, "J %d", mode);
            else std::snprintf(name, sizeof name, "tile_cols(%d) -> %d", -mode, opt);
            std::printf("    %-22s gate/up %7.2f ms, down %7.2f ms, both %7.2f (%5.2fx)  differ %d / %d%s\n", name, best_g,
                        best_d, best_g + best_d, t_def / (best_g + best_d), dh, dout, finite ? "" : "  NON-FINITE");
        }
        cudaFree(hq); cudaFree(out); cudaFree(gu); cudaFree(xq); cudaFree(x); cudaFree(ids); cudaFree(bounds);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("glm_mmq_glu_test: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench(s);
    if (argc > 1 && std::strcmp(argv[1], "--tiles") == 0) return tiles(s);

    // Two expert sets: 288 experts of 0-64 rows (Maya-L's count; the plain product's launch then has a block per
    // tile, as at a prompt chunk's size, and both sum every output over K in the same order: BIT FOR BIT), and six
    // experts of uneven rows - an empty one, a one-row one, one past a column tile - few enough tiles that the plain
    // product splits K over its blocks (stream-k): summed in another order there, so equal within float rounding.
    const int n_embd = 1024, n_ff = 256;
    std::mt19937 rng(20261013);
    std::vector<int32_t> many(288);
    for (auto& r : many) r = (int32_t) (rng() % 65);
    const std::vector<int32_t> edge = {37, 0, 130, 1, 64, 9};

    const int types[] = {GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                         GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
                         GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};
    mmq::Context ctx;
    int failures = 0, ran = 0;
    const std::vector<int32_t>* sets[] = {&many, &edge};
    for (const auto* set : sets) {
        const auto& rows = *set;
        const int n_exp = (int) rows.size();
        std::vector<int32_t> hb(n_exp + 1, 0);
        for (int e = 0; e < n_exp; ++e) hb[e + 1] = hb[e] + rows[e];
        const int T = hb[n_exp];
        const int max_rows = *std::max_element(rows.begin(), rows.end());
        // the plain product's stream-k launch: one block per tile when that fills >= 90% of the waves (mmq.cuh) - J
        // as mul_mat_q_switch_J picks it (the fewest column tiles), I = 128 rows
        int J = 8;
        while (J < 128 && (max_rows + J - 1) / J > (max_rows + 127) / 128) J += 8;
        const int ntiles = (max_rows + J - 1) / J * (2 * n_ff / 128) * n_exp, nsm = prop.multiProcessorCount;
        const bool exact = 100 * ntiles / (nsm * ((ntiles + nsm - 1) / nsm)) >= 90;
        std::printf("  %d experts, %d rows: the plain product %s\n", n_exp, T,
                    exact ? "tiles (expect bit for bit)" : "splits K (expect float rounding)");

        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> hx((size_t) T * n_embd);
        for (auto& v : hx) v = nd(rng);
        float* x = dev<float>(hx.size());
        check(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        int32_t* bounds = dev<int32_t>(hb.size());
        check(cudaMemcpy(bounds, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice), "bounds");
        int32_t* ids = dev<int32_t>(T);
        mmq::iota(ids, T, s);
        float* ga = dev<float>((size_t) T * 2 * n_ff);
        float* gf = dev<float>((size_t) T * 2 * n_ff);
        std::vector<float> ha((size_t) T * 2 * n_ff), hf((size_t) T * 2 * n_ff);
        for (int type : types) {
            if (!mmq::glu_supported(type, n_ff)) {
                std::printf("  %-8s no fused product on this device/build - skipped\n", ggml_type_name((ggml_type) type));
                continue;
            }
            // six quantized experts, repeated over the set (ggml's i-quant quantizers are slow)
            const size_t eb = mmq::matrix_bytes(type, 2 * n_ff, n_embd);
            const auto six = make_experts(type, 6, n_embd, n_ff, rng);
            std::vector<uint8_t> hw(eb * (size_t) n_exp);
            for (int e = 0; e < n_exp; ++e) std::memcpy(hw.data() + eb * e, six.data() + eb * (e % 6), eb);
            uint8_t* w = dev<uint8_t>(hw.size());
            check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "weights");
            uint8_t* xq = dev<uint8_t>(mmq::q8_bytes(T, n_embd));
            mmq::quantize(x, nullptr, xq, type, n_embd, n_embd, T, s);
            const Case c{w, eb, type, n_exp, n_embd, n_ff, T, max_rows, xq, bounds, ids};
            for (const float limit : {INFINITY, 0.75f}) {
                check(cudaMemsetAsync(ga, 0xff, (size_t) T * 2 * n_ff * 4, s), "memset");
                check(cudaMemsetAsync(gf, 0xff, (size_t) T * 2 * n_ff * 4, s), "memset");
                ctx.run(product(c, ga), s);
                gb::swiglu_rows(ga, ga, T, n_ff, limit, s, 2 * n_ff);
                ctx.run_glu(product(c, gf), limit, s);
                check(cudaMemcpyAsync(ha.data(), ga, ha.size() * 4, cudaMemcpyDeviceToHost, s), "download");
                check(cudaMemcpyAsync(hf.data(), gf, hf.size() * 4, cudaMemcpyDeviceToHost, s), "download");
                check(cudaStreamSynchronize(s), "sync");
                int differ = 0, far = 0, nonzero = 0;
                float worst = 0.f;
                for (int r = 0; r < T; ++r)
                    for (int k = 0; k < n_ff; ++k) {
                        const float a = ha[(size_t) r * 2 * n_ff + k], f = hf[(size_t) r * 2 * n_ff + k];
                        if (std::memcmp(&a, &f, 4) != 0) {
                            ++differ;
                            worst = std::max(worst, std::fabs(a - f));
                            far += !(std::fabs(a - f) <= 1e-4f * (1.f + std::fabs(a)));   // (NaN: far)
                        }
                        nonzero += a != 0.f;
                    }
                const bool ok = (exact ? differ == 0 : far == 0) && nonzero >= T * n_ff / 2;
                std::printf("  %-8s limit %-5g %7d values, %d differ (worst %.3g), %d nonzero%s\n",
                            ggml_type_name((ggml_type) type), limit, T * n_ff, differ, worst, nonzero, ok ? "" : "  FAIL");
                failures += !ok;
                ++ran;
            }
            cudaFree(xq);
            cudaFree(w);
        }
        cudaFree(gf);
        cudaFree(ga);
        cudaFree(ids);
        cudaFree(bounds);
        cudaFree(x);
    }
    if (failures != 0) {
        std::printf("glm_mmq_glu_test: %d FAILED\n", failures);
        return 1;
    }
    std::printf("glm_mmq_glu_test: %s\n", ran ? "all pass" : "skipped (no fused product here)");
    return 0;
}
