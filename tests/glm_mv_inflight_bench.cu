// tests/glm_mv_inflight_bench.cu - the dense GEMV with U super-blocks in flight (glm_fast.cu's mv_rdna kernels, on
// NVIDIA with STRATA_GLM_MV_INFLIGHT=1) against the original kernels (mv_kernel_t, the generic BF16 kernel, mv_nt's)
// at Maya's real shapes: synthetic Q6_K and BF16 weights (each case cycles enough copies to defeat the L2), the
// engine's job batches, one row (mv) and the rows of a batched decode step (mv_nt).  The NVIDIA twin of
// tests/hip/glm_mv_bench.cu.  Run from the build directory:
//   glm_mv_inflight_bench                 parity of the default against the original kernels, then the timings
//   glm_mv_inflight_bench --parity-only   bit-for-bit parity of every configuration only (a ctest)
//   glm_mv_inflight_bench --sweep         every rows/unroll/waves configuration
//   glm_mv_inflight_bench --case <name>   one case (kda_proj, kda_out, dsa_proj, dsa_q, dsa_out, router_shexp,
//                                         dense_gu, dense_down, head)
//   glm_mv_inflight_bench --nt 1,2,4      the activation counts (default 1,2,4)
#include "strata/kernels/glm_expert_bench.hpp"
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
uint16_t half_bits(float f) {
    const __half h = __float2half(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

struct Shape {
    int type, n_in, n_out;
};
struct Case {
    const char* name;
    std::vector<Shape> jobs;
};
// the engine's batches (glm_fast_path.cu / glm_mtp.cu), Maya-S v2 types
const std::vector<Case> kCases = {
    {"kda_proj", {{14, 4096, 8192}, {14, 4096, 8192}, {14, 4096, 8192}, {30, 4096, 128}, {30, 4096, 128}, {30, 4096, 64}}},
    {"kda_out", {{14, 8192, 4096}}},
    {"dsa_proj", {{14, 4096, 1536}, {14, 4096, 512}, {30, 4096, 128}, {30, 4096, 128}, {30, 4096, 32}}},
    {"dsa_q", {{14, 1536, 16384}, {30, 1536, 4096}}},
    {"dsa_out", {{14, 16384, 4096}}},
    {"router_shexp", {{30, 4096, 288}, {14, 4096, 2048}, {14, 4096, 2048}}},
    {"dense_gu", {{14, 4096, 12288}, {14, 4096, 12288}}},
    {"dense_down", {{14, 12288, 4096}}},
    {"head", {{14, 4096, 154880}}},
};
size_t wbytes(const Shape& s) { return gf::row_bytes(s.type, s.n_in) * (size_t) s.n_out; }

void fill_weights(std::vector<uint8_t>& v, const Shape& s, std::mt19937& rng) {
    v.resize(wbytes(s));
    if (s.type == 14) {
        for (size_t i = 0; i < v.size(); i += sizeof(block_q6_K)) {
            for (size_t j = 0; j < sizeof(block_q6_K); ++j) v[i + j] = rng() & 255;
            const float m = (1 + (rng() % 1024)) / 65536.f;
            const uint16_t d = half_bits((rng() & 1) ? m : -m);
            std::memcpy(v.data() + i + offsetof(block_q6_K, d), &d, 2);
        }
    } else {   // BF16 ~ N(0, 0.05)
        std::normal_distribution<float> nd(0.f, 0.05f);
        for (size_t i = 0; i < v.size(); i += 2) {
            const float fv = nd(rng);
            uint32_t b;
            std::memcpy(&b, &fv, 4);
            const uint16_t h = (uint16_t) (b >> 16);
            std::memcpy(v.data() + i, &h, 2);
        }
    }
}
// nt rows of n values: f32 and their q8_1
void fill_act(int n, int nt, std::mt19937& rng, std::vector<float>& xf, std::vector<uint8_t>& xq) {
    std::normal_distribution<float> nd(0.f, 1.f);
    xf.resize((size_t) n * nt);
    for (auto& f : xf) f = nd(rng);
    xq.resize((size_t) n / 32 * nt * sizeof(block_q8_1));
    for (size_t b = 0; b < xf.size() / 32; ++b) {
        float amax = 0, sum = 0;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(xf[32 * b + i])); sum += xf[32 * b + i]; }
        const float d = amax / 127.f;
        uint8_t* p = xq.data() + b * sizeof(block_q8_1);
        const uint16_t dh = half_bits(d), sh = half_bits(sum);
        std::memcpy(p, &dh, 2);
        std::memcpy(p + 2, &sh, 2);
        for (int i = 0; i < 32; ++i) p[4 + i] = uint8_t(int8_t(d == 0 ? 0 : std::lround(xf[32 * b + i] / d)));
    }
}

struct Config {
    int r, u, wpb, lds;
    std::string label() const {
        if (r == 0) return "original";
        if (r < 0) return "default";
        return "r" + std::to_string(r) + "u" + std::to_string(u) + "w" + std::to_string(wpb) + (lds ? "L" : "");
    }
};

// a pure read of `bytes` (uint4 loads, U in flight per lane): what the memory gives a streaming kernel
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, unsigned* out) {
    uint32_t acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n16; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n16; i += stride) acc ^= p[i].x ^ p[i].y ^ p[i].z ^ p[i].w;
    if (acc == 0x12345678u) out[0] = acc;
}

double time_us(cudaStream_t s, int calls, const std::function<void(int)>& launch) {
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    for (int i = 0; i < 4; ++i) launch(i);
    check(cudaStreamSynchronize(s), "warmup");
    std::vector<double> v;
    for (int rnd = 0; rnd < 7; ++rnd) {
        check(cudaEventRecord(a, s), "record");
        for (int i = 0; i < calls; ++i) launch(i);
        check(cudaEventRecord(b, s), "record");
        check(cudaEventSynchronize(b), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, a, b), "elapsed");
        v.push_back(ms * 1000.0 / calls);
    }
    std::sort(v.begin(), v.end());
    check(cudaEventDestroy(a), "event");
    check(cudaEventDestroy(b), "event");
    return v[v.size() / 2];
}

bool run_case(const Case& c, const std::vector<Config>& cfgs, const std::vector<int>& nts, bool parity_only,
              cudaStream_t s) {
    size_t bytes = 0;
    int max_in = 0, max_out = 0;
    for (const Shape& j : c.jobs) {
        bytes += wbytes(j);
        max_in = std::max(max_in, j.n_in);
        max_out = std::max(max_out, j.n_out);
    }
    // enough copies that a timed pass streams >= ~256 MB (a 48-128 MB L2 holds none of it)
    const int copies = parity_only ? 1 : (int) std::max<size_t>(1, std::min<size_t>(16, (256u << 20) / bytes + 1));
    std::mt19937 rng(0x5eed ^ (uint32_t) bytes);
    std::vector<void*> wdev(c.jobs.size() * copies);
    std::vector<uint8_t> h;
    for (size_t j = 0; j < c.jobs.size(); ++j)
        for (int k = 0; k < copies; ++k) {
            fill_weights(h, c.jobs[j], rng);
            check(cudaMalloc(&wdev[j * copies + k], h.size()), "cudaMalloc w");
            check(cudaMemcpy(wdev[j * copies + k], h.data(), h.size(), cudaMemcpyHostToDevice), "H2D w");
        }
    const int ntmax = *std::max_element(nts.begin(), nts.end());
    // each job its own activation (the engine's jobs share one, which only helps the caches)
    std::vector<void*> xq(c.jobs.size()), xf(c.jobs.size()), y(c.jobs.size()), yref(c.jobs.size());
    for (size_t j = 0; j < c.jobs.size(); ++j) {
        std::vector<float> f;
        std::vector<uint8_t> q;
        fill_act(c.jobs[j].n_in, ntmax, rng, f, q);
        check(cudaMalloc(&xq[j], q.size()), "cudaMalloc xq");
        check(cudaMemcpy(xq[j], q.data(), q.size(), cudaMemcpyHostToDevice), "H2D xq");
        check(cudaMalloc(&xf[j], f.size() * 4), "cudaMalloc xf");
        check(cudaMemcpy(xf[j], f.data(), f.size() * 4, cudaMemcpyHostToDevice), "H2D xf");
        check(cudaMalloc(&y[j], (size_t) c.jobs[j].n_out * ntmax * 4), "cudaMalloc y");
        check(cudaMalloc(&yref[j], (size_t) c.jobs[j].n_out * ntmax * 4), "cudaMalloc y");
    }
    // one row: mv (the decode's path, mv_kernel_t + the generic BF16 kernel); nt rows: mv_nt (a batched step's)
    auto launch = [&](std::vector<gf::MvJob>& J, int nt) {
        return nt == 1 ? gf::mv(J.data(), (int) J.size(), s) : gf::mv_nt(J.data(), (int) J.size(), nt, s);
    };
    auto jobs_of = [&](int copy, bool ref) {
        std::vector<gf::MvJob> J(c.jobs.size());
        for (size_t j = 0; j < c.jobs.size(); ++j) {
            const Shape& sh = c.jobs[j];
            J[j] = {wdev[j * copies + copy], sh.type == 14 ? xq[j] : nullptr, sh.type == 30 ? (const float*) xf[j] : nullptr,
                    (float*) (ref ? yref[j] : y[j]), nullptr, 1.0f, sh.type, sh.n_in, sh.n_out};
        }
        return J;
    };
    bool ok = true;
    for (int nt : nts) {
        // reference: the original kernels (timed first: the timing cycles the copies through yref, the parity
        // reference below is copy 0's)
        gf::hip_expert_bench::mv_config(0, 1, 8, 0);
        double t_ref = 0;
        if (!parity_only)
            t_ref = time_us(s, 20, [&](int i) {
                auto J = jobs_of(i % copies, true);
                launch(J, nt);
            });
        {
            auto J = jobs_of(0, true);
            if (!launch(J, nt)) { std::fprintf(stderr, "mv failed\n"); return false; }
            check(cudaStreamSynchronize(s), "ref");
        }
        for (const Config& cf : cfgs) {
            if (cf.r == 0) continue;
            // the window kernels exist for r1 u1/u2/u4 and r2 u1 (glm_fast.cu mv_rdna_launch)
            if (nt > 1 && cf.r > 0 && !(cf.r == 1 || (cf.r == 2 && cf.u == 1))) continue;
            gf::hip_expert_bench::mv_config(cf.r, cf.u, cf.wpb, cf.lds);
            for (size_t j = 0; j < c.jobs.size(); ++j) check(cudaMemset(y[j], 0xff, (size_t) c.jobs[j].n_out * nt * 4), "memset");
            auto J = jobs_of(0, false);
            launch(J, nt);
            check(cudaStreamSynchronize(s), "cfg");
            size_t diff = 0, total = 0;
            double max_abs = 0;
            for (size_t j = 0; j < c.jobs.size(); ++j) {
                const size_t n = (size_t) c.jobs[j].n_out * nt;
                std::vector<float> a(n), b(n);
                check(cudaMemcpy(a.data(), yref[j], n * 4, cudaMemcpyDeviceToHost), "D2H");
                check(cudaMemcpy(b.data(), y[j], n * 4, cudaMemcpyDeviceToHost), "D2H");
                for (size_t i = 0; i < n; ++i) {
                    if (std::memcmp(&a[i], &b[i], 4) != 0) ++diff;
                    if (std::isfinite(a[i]) && std::isfinite(b[i])) max_abs = std::max(max_abs, (double) std::fabs(a[i] - b[i]));
                    else if (std::isfinite(a[i]) != std::isfinite(b[i])) max_abs = INFINITY;
                }
                total += n;
            }
            const bool exact = diff == 0;
            ok &= exact;
            if (parity_only) {
                std::printf("parity %-13s nt=%d %-9s outputs=%zu differ=%zu max_abs=%.3e %s\n", c.name, nt,
                            cf.label().c_str(), total, diff, max_abs, exact ? "PASS" : "FAIL");
                continue;
            }
            const double t = time_us(s, 20, [&](int i) {
                auto Jt = jobs_of(i % copies, false);
                launch(Jt, nt);
            });
            std::printf("bench %-13s nt=%d %-9s %8.1f us %6.1f GB/s | original %8.1f us %6.1f GB/s | %.3fx %s\n", c.name,
                        nt, cf.label().c_str(), t, bytes / t / 1000.0, t_ref, bytes / t_ref / 1000.0, t_ref / t,
                        exact ? "bit-exact" : "DIFFERS");
        }
    }
    gf::hip_expert_bench::mv_config(-1, 2, 8, -1);
    for (void* p : wdev) check(cudaFree(p), "free");
    for (size_t j = 0; j < c.jobs.size(); ++j) {
        check(cudaFree(xq[j]), "free");
        check(cudaFree(xf[j]), "free");
        check(cudaFree(y[j]), "free");
        check(cudaFree(yref[j]), "free");
    }
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    bool parity_only = false, sweep = false;
    std::string only;
    std::vector<int> nts = {1, 2, 4};
    // the in-flight kernels on NVIDIA are opt-in: on for this process (mv_config(0, ...) gives the originals)
    setenv("STRATA_GLM_MV_INFLIGHT", "1", 1);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--parity-only") parity_only = true;
        else if (a == "--sweep") sweep = true;
        else if (a == "--case" && i + 1 < argc) only = argv[++i];
        else if (a == "--nt" && i + 1 < argc) {
            nts.clear();
            for (const char* p = argv[++i]; *p;) {
                nts.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
        else {
            std::fprintf(stderr, "usage: %s [--parity-only] [--sweep] [--case name] [--nt 1,2,4]\n", argv[0]);
            return 2;
        }
    }
    check(cudaSetDevice(0), "GPU 0");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "device properties");
    std::printf("GPU 0: %s (cc %d.%d), %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    if (!parity_only) {   // the streaming roof
        const size_t n = 512u << 20;
        void* p;
        unsigned* o;
        check(cudaMalloc(&p, n), "cudaMalloc read");
        check(cudaMalloc(&o, 4), "cudaMalloc o");
        check(cudaMemset(p, 1, n), "memset");
        for (int blocks : {prop.multiProcessorCount * 8, prop.multiProcessorCount * 16, prop.multiProcessorCount * 32}) {
            const double t = time_us(s, 5, [&](int) { read_kernel<<<blocks, 256, 0, s>>>((const uint4*) p, n / 16, o); });
            std::printf("roof  read 512 MiB blocks=%d: %.1f us %.1f GB/s\n", blocks, t, n / t / 1000.0);
        }
        check(cudaFree(p), "free");
        check(cudaFree(o), "free");
    }
    std::vector<Config> cfgs = {{-1, 2, 8, -1}};
    if (sweep || parity_only) {
        cfgs.clear();
        for (int lds : {0, 1})
            for (int wpb : {8, 4})
                for (int r : {1, 2, 4})
                    for (int u : {1, 2, 4}) cfgs.push_back({r, u, wpb, lds});
        cfgs.push_back({-1, 2, 8, -1});   // and the default (its LDS choice)
    }
    bool ok = true;
    for (const Case& c : kCases) {
        if (!only.empty() && only != c.name) continue;
        std::vector<Config> cc = cfgs;
        ok &= run_case(c, cc, nts, parity_only, s);
    }
    check(cudaStreamDestroy(s), "stream");
    std::printf("glm_mv_inflight_bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
