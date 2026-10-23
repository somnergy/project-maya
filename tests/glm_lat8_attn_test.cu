// tests/glm_lat8_attn_test.cu - the tensor-core decode attention (gf::mla_attend_batch) over INT8 latent records against
// the same attention over FP16 rows holding each record's values ((half) (q * d), the conversion the kernel does),
// bitwise (needs an NVIDIA GPU of compute capability 8.0+, no model).  Run it as is (the INT8 records read a byte at a
// time) and with STRATA_GLM_ATTN_LAT8_VEC=1 (8 values a load): both must give the FP16 rows' bits.  It also prints each
// path's time for one token and for a launch of 4 (2,051 cells each, the model's selection).
#include "strata/kernels/glm_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// GLM-5.x: 64 heads, qk_nope 256, kv_lora 512; 512 pools of 4 cells selected (+ the tail)
constexpr int kHead = 64, kNope = 256, kLora = 512, kNSelMax = 512 * 4 + 3;
constexpr int kRec = kLora + kLora / 16;
constexpr int kPos = 1 << 16;
constexpr int kTok = 4;

}  // namespace

int main() {
    int dev = 0, major = 0;
    ck(cudaGetDevice(&dev), "device");
    ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc");
    if (major < 8) {
        std::printf("SKIP: needs compute capability 8.0+\n");
        return 77;
    }
    const char* vec = std::getenv("STRATA_GLM_ATTN_LAT8_VEC");
    std::printf("INT8 records read %s\n", vec != nullptr && std::atoi(vec) != 0 ? "8 values a load" : "a value a load");
    std::mt19937 rng(20261023u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the records: codes from a normal spread, scales as the cache writes them (amax / 127 in FP16); some blocks of
    // zeros (scale 0), some near-zero blocks (subnormal scales), a few negative-zero / negative scales for the signs
    std::vector<uint8_t> rec((size_t) kPos * kRec);
    std::vector<uint16_t> row((size_t) kPos * kLora);
    for (int p = 0; p < kPos; ++p) {
        uint8_t* r = rec.data() + (size_t) p * kRec;
        for (int c = 0; c < kLora; ++c)
            r[c] = (uint8_t) (int8_t) std::clamp((int) std::lround(nd(rng) * 40.0f), -127, 127);
        for (int b = 0; b < kLora / 32; ++b) {
            const uint32_t k = rng() % 1000;
            float d = std::exp2(nd(rng) * 2.0f - 6.0f);
            if (k < 5) d = 0.0f;
            else if (k < 10) d = std::ldexp(1.0f, -20 - (int) (rng() % 4));   // FP16 subnormal
            else if (k < 12) d = -d;
            else if (k < 13) d = -0.0f;
            const __half h = __float2half(d);
            std::memcpy(r + kLora + 2 * b, &h, 2);
        }
        for (int c = 0; c < kLora; ++c) {
            __half h;
            std::memcpy(&h, r + kLora + 2 * (c / 32), 2);
            const __half v = __float2half((float) (int8_t) r[c] * __half2float(h));
            std::memcpy(&row[(size_t) p * kLora + c], &v, 2);
        }
    }
    uint8_t* d_rec = dalloc<uint8_t>(rec.size());
    uint16_t* d_row = dalloc<uint16_t>(row.size());
    ck(cudaMemcpy(d_rec, rec.data(), rec.size(), cudaMemcpyHostToDevice), "h2d");
    ck(cudaMemcpy(d_row, row.data(), row.size() * 2, cudaMemcpyHostToDevice), "h2d");
    float* d_q = dalloc<float>((size_t) kTok * kHead * kLora);
    int* d_cells = dalloc<int>((size_t) kTok * kNSelMax);
    float* d_ref = dalloc<float>((size_t) kTok * kHead * kLora);
    float* d_got = dalloc<float>((size_t) kTok * kHead * kLora);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const size_t ctx_n = (size_t) kTok * kHead * kLora;
    std::vector<float> a(ctx_n), b(ctx_n);
    int fail = 0, checks = 0;
    // cell counts: one chunk's worth and less, a block's split boundary, the model's full selection
    const int nsels[] = {1, 7, 31, 32, 33, 100, 511, 1000, 2047, kNSelMax};
    for (int round = 0; round < 3; ++round) {
        for (int ns : nsels) {
            for (int T : {1, kTok}) {
                std::vector<float> q((size_t) kTok * kHead * kLora);
                for (float& x : q) x = nd(rng) * 0.5f;
                std::vector<int> cells((size_t) kTok * kNSelMax, -1), nsel(kTok);
                for (int t = 0; t < kTok; ++t) {
                    nsel[t] = std::max(1, ns - (t * (int) (rng() % 64)));
                    for (int i = 0; i < nsel[t]; ++i) cells[(size_t) t * kNSelMax + i] = (int) (rng() % kPos);
                    if (round == 2)   // some padding inside the list (a selection shorter than its pools)
                        for (int i = 0; i < nsel[t]; i += 37) cells[(size_t) t * kNSelMax + i] = -1;
                }
                ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "h2d");
                ck(cudaMemcpy(d_cells, cells.data(), cells.size() * 4, cudaMemcpyHostToDevice), "h2d");
                ck(cudaMemset(d_ref, 0xff, ctx_n * 4), "memset");
                ck(cudaMemset(d_got, 0xee, ctx_n * 4), "memset");
                const int64_t qs = (int64_t) kHead * kLora;
                if (!gf::mla_attend_batch(d_q, qs, d_row, d_cells, kNSelMax, nsel.data(), T, kHead, kNope, kLora, d_ref,
                                          qs, s, false) ||
                    !gf::mla_attend_batch(d_q, qs, (const uint16_t*) d_rec, d_cells, kNSelMax, nsel.data(), T, kHead,
                                          kNope, kLora, d_got, qs, s, true)) {
                    std::printf("FAIL: mla_attend_batch not available\n");
                    return 1;
                }
                ck(cudaStreamSynchronize(s), "sync");
                ck(cudaMemcpy(a.data(), d_ref, (size_t) T * qs * 4, cudaMemcpyDeviceToHost), "d2h");
                ck(cudaMemcpy(b.data(), d_got, (size_t) T * qs * 4, cudaMemcpyDeviceToHost), "d2h");
                ++checks;
                if (std::memcmp(a.data(), b.data(), (size_t) T * qs * 4) != 0) {
                    size_t i = 0;
                    while (std::memcmp(&a[i], &b[i], 4) == 0) ++i;
                    std::printf("FAIL: n_sel %d, %d tokens, round %d: ctx[%zu] %.9g (FP16 rows) vs %.9g (INT8)\n", ns, T,
                                round, i, a[i], b[i]);
                    ++fail;
                }
            }
        }
    }
    // time: one token and 4 at the model's selection, FP16 rows vs INT8 records
    {
        std::vector<int> cells((size_t) kTok * kNSelMax), nsel(kTok, kNSelMax);
        for (int& c : cells) c = (int) (rng() % kPos);
        ck(cudaMemcpy(d_cells, cells.data(), cells.size() * 4, cudaMemcpyHostToDevice), "h2d");
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        const int64_t qs = (int64_t) kHead * kLora;
        for (int T : {1, kTok}) {
            for (int q8 = 0; q8 < 2; ++q8) {
                const uint16_t* lat = q8 ? (const uint16_t*) d_rec : d_row;
                for (int w = 0; w < 5; ++w)
                    gf::mla_attend_batch(d_q, qs, lat, d_cells, kNSelMax, nsel.data(), T, kHead, kNope, kLora, d_got,
                                         qs, s, q8 != 0);
                const int reps = 100;
                ck(cudaEventRecord(e0, s), "record");
                for (int w = 0; w < reps; ++w)
                    gf::mla_attend_batch(d_q, qs, lat, d_cells, kNSelMax, nsel.data(), T, kHead, kNope, kLora, d_got,
                                         qs, s, q8 != 0);
                ck(cudaEventRecord(e1, s), "record");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0.0f;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                std::printf("time: %d token(s), %s: %.1f us a call (attention + merge)\n", T,
                            q8 ? "INT8 records" : "FP16 rows", ms * 1000.0f / reps);
            }
        }
    }
    std::printf("%s: %d of %d cases bitwise equal\n", fail ? "FAIL" : "OK", checks - fail, checks);
    return fail ? 1 : 0;
}
