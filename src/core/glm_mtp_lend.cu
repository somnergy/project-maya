// src/core/glm_mtp_lend.cu - STRATA_GLM_MTP_LEND=1 (opt-in; needs STRATA_GLM_MULTI_MTP on a split): the NextN draft
// layer's expert memory serves the trunk while drafting is off, and comes back for drafting.
//
// The draft layer sits on the split's last part, and its MAIN slots are one block of that part's expert pool
// (lp[mtp_il_].base, n_main * stride).  With several conversations the scheduler drafts only while
// STRATA_GLM_MULTI_MTP_MAX or fewer run (generate.cpp), and a lone request drafts in its speculative decode; with more
// running that block holds experts nobody reads.  Here:
//   - mtp_lend_setup (fast_setup, after the pool's layout): the block's bytes as BORROWED slots of this part's trunk
//     MoE layers (LayerPool slots [n_tail, n) at bbase) - how many each by the pool's own split continued
//     (strata/core/mtp_lend.hpp, mtp_lend_counts), laid out layer-major (tier_tail_layout).  They start kLent: the
//     warm-up loads the draft layer as before.  A pinned copy of the block's size is kept for the way back.
//   - lend (mtp_lend_part(true)): the tier moves retire (as lend_tail's), the draft layer's main slots leave the tables
//     (its slot map is remembered), the block's bytes go to the pinned copy, the draft's main slots become kLent and the
//     borrowed ones kFree.  Filling them is the boundary's own work: spares from free slots, and the refill that
//     promotes each layer's hottest RAM-tier experts into its free slots (STRATA_GLM_PROMOTE_FILL a boundary).
//   - reclaim (mtp_lend_part(false)): the borrowed slots empty by the prompt's lending rules (lend_tail's borrowed
//     flag: an expert there takes a colder main resident's slot, that one parked in the RAM tier, or is parked itself),
//     the pinned copy goes back into the block, and the draft layer's slot map and table entries are restored as they
//     were - the same experts in the same slots, byte for byte, so its drafts are what they would have been.
// While lent the boundary leaves the draft layer alone (no spares, no moves), so nothing can move a remembered expert
// elsewhere, and no draft runs (spec_ready() is false; the scheduler's windows wait for mtp_lent() to clear).  Both
// switches are synchronous on this part, called with every part idle (the scheduler drains first).
#include "glm_fast_state.hpp"
#include "glm_rows_state.hpp"

#include "strata/core/mtp_lend.hpp"
#include "strata/core/tier_policy.hpp"
#include "strata/kernels/glm_batch.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

namespace {
// the device tables' edits in batches (each key at most once a batch), as the boundary's
using FastState = Glm5Model::FastState;
struct TableEdits {
    FastState* F;
    std::string& err;
    int nu = 0;
    bool ok = true;
    bool flush() {
        if (nu == 0 || !ok) return ok;
        ok = glmfast::cuda_ok(cudaMemcpyAsync(F->upd_key_d, F->upd_key_h, (size_t) nu * sizeof(int),
                                              cudaMemcpyHostToDevice, F->cs), "glm mtp lend keys", err) &&
             glmfast::cuda_ok(cudaMemcpyAsync(F->upd_val_d, F->upd_val_h, (size_t) nu * sizeof(unsigned long long),
                                              cudaMemcpyHostToDevice, F->cs), "glm mtp lend values", err);
        if (ok) {
            gf::tab_update(F->tab, F->upd_key_d, F->upd_val_d, nu, F->cs);
            ok = glmfast::cuda_ok(cudaGetLastError(), "glm mtp lend table update", err);
        }
        const cudaError_t e = cudaStreamSynchronize(F->cs);   // (the pinned sources are reused after it)
        if (ok) ok = glmfast::cuda_ok(e, "glm mtp lend table sync", err);
        nu = 0;
        return ok;
    }
    void upd(size_t k, unsigned long long v) {
        if (!ok || (nu == FastState::kMaxUpd && !flush())) return;
        F->upd_key_h[nu] = (int) k;
        F->upd_val_h[nu] = v;
        ++nu;
    }
};

double secs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
}  // namespace

bool Glm5Model::mtp_lend_setup(const std::vector<std::vector<double>>& share, std::string& err) {
    (void) err;
    FastState* F = fast_;
    auto& M = F->ml;
    const Glm5Geometry& g = g_;
    const MtpLendExtra& X = mtp_lend_extra();
    const int d = mtp_il_;
    const bool draft = d >= l0_ && d < lt_ && F->L[(size_t) d].moe;   // (this part carries the draft layer)
    // STRATA_GLM_MTP_LEND_MORE: a part without it lends its windows' buffers (a model with the NextN block loaded)
    if (!draft && !(X.more && g.nextn > 0 && getenv("STRATA_GLM_NO_MTP") == nullptr)) return true;
    const char* why = nullptr;
    if (multi_mtp_setting() <= 0) why = "it needs MAYA_GLM_MULTI_MTP";
    else if (n_parts_ < 2 || (draft && split_next_ != nullptr)) why = "it needs a split (the draft layer on its last part)";
    else if (seq_count() < 2 || split_rows_setting() <= 0)
        why = "it needs --parallel and MAYA_GLM_SPLIT_ROWS (the windows' scheduler decides when)";
    else if (F->unified_memory || F->complete) why = "the pool holds every expert already";
    else if (draft && F->lp[(size_t) d].n_main <= 0) why = "the draft layer has no main slots";
    if (why != nullptr) {
        std::fprintf(stderr, "glm mtp lend: CUDA%d off - %s\n", dev_, why);
        return true;
    }
    // the regions: the draft layer's main slots (its bytes copied out at each lend), then (MORE) the windows' block
    // (scratch) and the NextN block's weights that nothing reads while lent (copied out once, now)
    std::vector<size_t> rb;
    std::vector<uint8_t*> rp;
    if (draft) {
        const auto& PD = F->lp[(size_t) d];
        M.il = d;
        M.region = PD.base;
        M.bytes = (size_t) PD.n_main * PD.stride;
        rp.push_back(M.region);
        rb.push_back(M.bytes);
    }
    M.more.clear();
    if (X.more) {
        const auto blk = mtpw_block();
        if (blk.first != nullptr) M.more.push_back({blk.first, blk.second, nullptr, "the windows' buffers"});
        // (the dense weights only when reads leave the cache fill out: a read is their one user while lent)
        if (draft && X.skip_read && mtp_arena_ != nullptr && mtp_arena_bytes_ > 0)
            M.more.push_back({(uint8_t*) mtp_arena_, (size_t) mtp_arena_bytes_, nullptr, "the NextN block's weights"});
        if (draft && X.skip_read && F->mtp_emb != nullptr)
            M.more.push_back({(uint8_t*) F->mtp_emb,
                              (size_t) g.n_vocab * ggml_row_size((ggml_type) F->mtp_emb_type, g.n_embd), nullptr,
                              "token_embd"});
        for (auto& r : M.more) {
            if (r.p == blk.first) continue;   // (scratch)
            if (cudaHostAlloc((void**) &r.copy, r.bytes, cudaHostAllocDefault) != cudaSuccess ||
                cudaMemcpy(r.copy, r.p, r.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                cudaGetLastError();
                if (r.copy != nullptr) cudaFreeHost(r.copy);
                r.copy = nullptr;
                r.bytes = 0;   // (not lent: it stays where it is)
                std::fprintf(stderr, "glm mtp lend: CUDA%d %s stay (no pinned copy)\n", dev_, r.what);
            }
        }
        M.more.erase(std::remove_if(M.more.begin(), M.more.end(), [](const FastState::MtpLend::More& r) { return r.bytes == 0; }),
                     M.more.end());
        for (const auto& r : M.more) {
            rp.push_back(r.p);
            rb.push_back(r.bytes);
        }
    }
    if (rb.empty()) return true;
    // this part's trunk MoE layers and how many borrowed slots each: the pool's split continued over them
    std::vector<int> ils, have;
    std::vector<size_t> strides;
    for (int il = l0_; il < lt_; ++il) {
        if (il == d || !F->L[(size_t) il].moe) continue;
        ils.push_back(il);
        strides.push_back(F->lp[(size_t) il].stride);
        have.push_back(F->lp[(size_t) il].n);
    }
    const auto gain = [&](int k, int h) -> double {
        const size_t il = (size_t) ils[(size_t) k];
        if (il >= share.size()) return 0.0;
        const int j = h - gf::kSpares;   // (the pool split's own gain: the next expert's share; spares serve none)
        return j >= 0 && j < (int) share[il].size() ? share[il][(size_t) j] : 0.0;
    };
    const std::vector<std::vector<MtpLendSlot>> pl = mtp_lend_place(rb, strides, have, g.n_expert + gf::kSpares, gain);
    int tot = 0, layers = 0, cmin = INT_MAX, cmax = 0, in_more = 0;
    for (size_t k = 0; k < ils.size(); ++k) {
        const int c = (int) pl[k].size();
        if (c <= 0) continue;
        tot += c;
        ++layers;
        cmin = std::min(cmin, c);
        cmax = std::max(cmax, c);
        for (const auto& sl : pl[k]) in_more += !draft || sl.region > 0;
    }
    if (tot == 0) {
        std::fprintf(stderr, "glm mtp lend: CUDA%d off - MTP's lendable memory holds no trunk slot\n", dev_);
        for (auto& r : M.more)
            if (r.copy != nullptr) cudaFreeHost(r.copy);
        M.more.clear();
        return true;
    }
    // the pinned copy the draft layer's bytes wait in while lent
    if (draft && cudaHostAlloc((void**) &M.snap, M.bytes, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        M.snap = nullptr;
        std::fprintf(stderr, "glm mtp lend: CUDA%d off - %.2f GB of pinned RAM for the draft layer's copy did not "
                             "allocate\n", dev_, (double) M.bytes / 1073741824.0);
        for (auto& r : M.more)
            if (r.copy != nullptr) cudaFreeHost(r.copy);
        M.more.clear();
        return true;
    }
    for (size_t k = 0; k < ils.size(); ++k) {
        if (pl[k].empty()) continue;
        auto& P = F->lp[(size_t) ils[k]];
        P.bptr.clear();
        for (const auto& sl : pl[k]) P.bptr.push_back(rp[(size_t) sl.region] + sl.off);
        P.n = P.n_tail + (int) pl[k].size();
        P.key.resize((size_t) P.n, -1);
        P.tick.resize((size_t) P.n, 0);
        P.st.resize((size_t) P.n, FastState::kLent);   // (MTP's memory until it is lent)
    }
    M.on = true;
    M.slots = tot;
    M.layers = layers;
    if (draft) {
        M.key.assign((size_t) F->lp[(size_t) d].n_main, -1);
        M.tick.assign((size_t) F->lp[(size_t) d].n_main, 0);
        mtp_catch_setup();
    }
    const MtpLendPolicy& pol = mtp_lend_policy();
    std::string more;
    size_t more_b = 0, more_pin = 0;
    for (const auto& r : M.more) {
        more += std::string(more.empty() ? "" : ", ") + r.what + " " + std::to_string((r.bytes + (1 << 19)) >> 20) + " MB";
        more_b += r.bytes;
        more_pin += r.copy != nullptr ? r.bytes : 0;
    }
    char head[256], tail[384] = "";
    if (draft)
        std::snprintf(head, sizeof head, "the draft layer's %d main slots (%.2f GB)", F->lp[(size_t) d].n_main,
                      (double) M.bytes / 1073741824.0);
    else
        std::snprintf(head, sizeof head, "MTP's memory (%s)", more.c_str());
    if (draft && !more.empty())
        std::snprintf(tail, sizeof tail, ", with MTP's other memory (%s: %d of the slots)", more.c_str(), in_more);
    std::fprintf(stderr, "glm mtp lend: CUDA%d %s serve %d slots of %d trunk layers (%d-%d a layer)%s while drafting is "
                         "off (MAYA_GLM_MTP_LEND%s): lent after %.0f s with more than MAYA_GLM_MULTI_MTP_MAX running, back "
                         "after %.0f s at that or fewer (one or none: at once); %.2f GB pinned for the copies (%.2f GB "
                         "lent in all)\n",
                 dev_, head, tot, layers, cmin, cmax, tail, more.empty() ? "" : ", _MORE", pol.lend_after,
                 pol.reclaim_after, (double) ((draft ? M.bytes : 0) + more_pin) / 1073741824.0,
                 (double) ((draft ? M.bytes : 0) + more_b) / 1073741824.0);
    return true;
}

// STRATA_GLM_MTP_LEND_SKIP_READ: reads leave the NextN cache fill out while lent; the pinned rows for the catch-up
void Glm5Model::mtp_catch_setup() {
    FastState* F = fast_;
    const MtpLendExtra& X = mtp_lend_extra();
    if (!X.skip_read || mtp_il_ < 0) return;
    auto& K = F->mcatch;
    K.on = true;
    K.W = X.catchup;
    const int ns = std::max(1, seq_count());
    const size_t E = (size_t) g_.n_embd;
    if (K.W > 0) {
        const size_t bytes = (size_t) ns * (size_t) K.W * E * sizeof(float);
        if (cudaHostAlloc((void**) &K.h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &K.h_d, K.h, 0) != cudaSuccess) {
            cudaGetLastError();
            if (K.h != nullptr) cudaFreeHost(K.h);
            K.h = K.h_d = nullptr;
            std::fprintf(stderr, "glm mtp lend: CUDA%d %.2f GB of pinned RAM for the catch-up rows did not allocate - the "
                                 "NextN cache entries reads leave out while lent stay as they were\n",
                         dev_, (double) bytes / 1073741824.0);
            K.W = 0;
        } else {
            K.ring.assign((size_t) ns, MtpCatchRing{});
            for (auto& r : K.ring) r.init(K.W);
        }
    }
    char tail[160];
    if (K.W > 0)
        std::snprintf(tail, sizeof tail, "the newest %d positions a conversation fill it before it drafts again "
                                         "(MAYA_GLM_MTP_LEND_CATCHUP, %.0f MB pinned)",
                      K.W, (double) ns * K.W * E * sizeof(float) / 1048576.0);
    else
        std::snprintf(tail, sizeof tail, "none filled later (MAYA_GLM_MTP_LEND_CATCHUP=0)");
    std::fprintf(stderr, "glm mtp lend: CUDA%d prompt reads leave the NextN cache fill out while its slots are lent "
                         "(MAYA_GLM_MTP_LEND_SKIP_READ); %s\n", dev_, tail);
}

bool Glm5Model::mtp_skip_fill() const {
    const FastState* F = fast_;
    return F != nullptr && F->mcatch.on && F->ml.on && F->ml.lent;
}

void Glm5Model::mtp_catch_drop(int sq, int64_t from) {
    FastState* F = fast_;
    if (F == nullptr || F->mcatch.W <= 0 || sq < 0 || sq >= (int) F->mcatch.ring.size()) return;
    F->mcatch.ring[(size_t) sq].drop_from(from);
}

// the chunk's rows [0, Tv) at p0 .. p0 + Tv - 1 had their NextN cache entries left out: the newest W of them keep the
// trunk's hidden state (head_rows', the NextN block's input) in the pinned rows - copies on this stream, so the
// catch-up's kernels on it read them after they landed
void Glm5Model::mtp_catch_keep(int64_t p0, int Tv, const int32_t* next_ids, const float* R, float* hbuf,
                               cudaStream_t s) {
    FastState* F = fast_;
    auto& K = F->mcatch;
    K.skipped += (uint64_t) Tv;
    if (K.W <= 0 || Tv <= 0 || cur_seq_ < 0 || cur_seq_ >= (int) K.ring.size()) return;
    auto& Q = K.ring[(size_t) cur_seq_];
    const int keep = std::min(Tv, K.W), t0 = Tv - keep;
    const size_t E = (size_t) g_.n_embd;
    strata::kernels::glmb::head_rows(R + (size_t) t0 * 4 * E, w_.at("output_norm.weight"), g_.norm_eps, keep,
                                     (int) E, hbuf, s);
    int seg[2][3];
    const int ns = Q.append(p0 + t0, keep, next_ids + t0, seg);
    for (int i = 0; i < ns; ++i)
        cudaMemcpyAsync(K.h + ((size_t) cur_seq_ * (size_t) K.W + (size_t) seg[i][0]) * E, hbuf + (size_t) seg[i][2] * E,
                        (size_t) seg[i][1] * E * sizeof(float), cudaMemcpyDeviceToHost, s);
}

// conversation sq's left-out entries into its NextN cache (this part carries the block, owned again): the window
// path's cache rows (mtpw_cache_rows: eh_proj, the attention norm, kv_a, the indexer's key and gate - KV streaming's
// hook where the cache is streamed), mtpc's rows at a time, oldest first.  The pinned rows are read in place.
bool Glm5Model::mtp_catchup_part(int sq, std::string& err) {
    FastState* F = fast_;
    if (F == nullptr) return true;
    auto& K = F->mcatch;
    if (!K.on || K.W <= 0 || sq < 0 || sq >= (int) K.ring.size() || K.ring[(size_t) sq].count == 0) return true;
    if (F->ml.lent || F->mtpc.cap < 1) return true;   // (the callers draft: owned - never expected)
    auto& Q = K.ring[(size_t) sq];
    const auto t0 = std::chrono::steady_clock::now();
    RowSeq rb;
    mtpw_bind(sq, rb);
    const size_t E = (size_t) g_.n_embd;
    int n_fill = 0;
    bool ok = true;
    Q.runs(rb.p, F->mtpc.cap, [&](int i0, int64_t p, int n) {
        if (!ok) return;
        // (mtpc's pinned embedding rows are reused: the previous run's copy out of them must have landed)
        ok = glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm mtp catch-up", err) &&
             mtpw_cache_rows(rb, p, n, K.h_d + ((size_t) sq * (size_t) K.W + (size_t) i0) * E, Q.next.data() + i0,
                             F->mtpc.emb_h, err);
        n_fill += n;
    });
    const uint64_t lost = Q.lost;
    Q.clear();
    Q.lost = 0;
    if (!ok || !glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm mtp catch-up", err)) return false;
    const double ms = 1000.0 * secs(t0, std::chrono::steady_clock::now());
    K.filled += (uint64_t) n_fill;
    ++K.fills;
    K.fill_ms += ms;
    std::fprintf(stderr, "glm mtp lend: CUDA%d conversation %d: %d NextN cache entries its reads left out while lent "
                         "filled in %.1f ms before it drafts (%llu older ones not kept)\n",
                 dev_, sq, n_fill, ms, (unsigned long long) lost);
    return true;
}

bool Glm5Model::mtp_catchup(int sq, std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || !m->fast_->mcatch.on) continue;
        cudaSetDevice(m->dev_);
        const bool ok = m->mtp_catchup_part(sq, err);
        cudaSetDevice(dev_);
        return ok;
    }
    return true;
}

void Glm5Model::mtp_lend_free() {
    FastState* F = fast_;
    if (F == nullptr) return;
    auto& K = F->mcatch;
    if (K.on && K.skipped > 0)
        std::fprintf(stderr, "glm mtp lend: CUDA%d reads left %llu NextN cache entries out while lent; %llu filled by "
                             "%llu catch-ups in %.0f ms\n", dev_, (unsigned long long) K.skipped,
                     (unsigned long long) K.filled, (unsigned long long) K.fills, K.fill_ms);
    if (K.h != nullptr) cudaFreeHost(K.h);
    K.h = K.h_d = nullptr;
    K.on = false;
    auto& M = F->ml;
    if (M.on && (M.lends > 0 || M.reclaims > 0)) {
        const auto now = std::chrono::steady_clock::now();
        const double lent = M.lent_s + (M.lent ? secs(M.t_lent, now) : 0.0), all = secs(M.t_start, now);
        std::fprintf(stderr, "glm mtp lend: CUDA%d %llu lends, %llu reclaims; lent %.0f of %.0f s (%.0f%%)\n", dev_,
                     (unsigned long long) M.lends, (unsigned long long) M.reclaims, lent, all,
                     100.0 * lent / std::max(1e-9, all));
    }
    if (M.snap != nullptr) cudaFreeHost(M.snap);
    M.snap = nullptr;
    for (auto& r : M.more)
        if (r.copy != nullptr) cudaFreeHost(r.copy);
    M.more.clear();
    M.on = false;
}

namespace {
// the trunk experts of layers [l0, l1) but `skip` in no tier (neither a VRAM slot nor a RAM one): read from the disk next
int64_t experts_out(const FastState* F, int l0, int l1, int skip, int NE) {
    int64_t out = 0;
    for (int il = l0; il < l1; ++il) {
        if (il == skip || !F->L[(size_t) il].moe) continue;
        for (int e = 0; e < NE; ++e) {
            const size_t key = (size_t) il * NE + e;
            out += F->slot_of[key] < 0 && F->ram_of[key] < 0;
        }
    }
    return out;
}
}  // namespace

// A failed switch leaves the tiers moving again (bg_hold off) and the draft layer marked lent for good: whatever state
// its slots are in, nothing drafts from them, and no switch is tried again (the error still ends the running requests).
bool Glm5Model::mtp_lend_part(bool lend, const std::string& why, std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || !F->ml.on || F->ml.broken || F->ml.lent == lend) return true;
    if (mtp_lend_step_(lend, why, err)) return true;
    F->bg_hold = false;
    F->ml.broken = true;
    F->ml.lent = true;
    std::fprintf(stderr, "glm mtp lend: CUDA%d the %s failed (%s): no more switches, and no drafting from here on\n", dev_,
                 lend ? "lend" : "reclaim", err.c_str());
    return false;
}

bool Glm5Model::mtp_lend_step_(bool lend, const std::string& why, std::string& err) {
    FastState* F = fast_;
    auto& M = F->ml;
    const int NE = g_.n_expert;
    const bool draft = M.il >= 0;   // (MAYA_GLM_MTP_LEND_MORE: a part without the draft layer lends its buffers only)
    static FastState::LayerPool none;
    auto& PD = draft ? F->lp[(size_t) M.il] : none;
    const auto t0 = std::chrono::steady_clock::now();
    kda_bak_cur_ = false;   // (the recurrent-state copies may be lent: never the live states' equal after this)
    if (lend) {
        // (1) the tier moves in flight land and retire, and none starts (lend_tail's way)
        F->bg_hold = true;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm mtp lend copies", err) || !fast_boundary(err))
            return false;
        if (!F->bg.empty() || !F->draining.empty()) {
            err = "glm mtp lend: tier moves were not retired";
            return false;
        }
        // (2) the draft layer's main slots leave the tables (their map remembered); a spare there leaves the spare
        //     table (it comes back as a free slot)
        int held = 0;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            TableEdits ed{F, err};
            for (int s = 0; s < PD.n_main; ++s) {
                char& st = PD.st[(size_t) s];
                M.key[(size_t) s] = -1;
                M.tick[(size_t) s] = PD.tick[(size_t) s];
                if (st == FastState::kResident && PD.key[(size_t) s] >= 0) {
                    const int key = M.il * NE + PD.key[(size_t) s];
                    M.key[(size_t) s] = PD.key[(size_t) s];
                    ed.upd(F->tab_key(key), 0ull);
                    if (F->slot_of[(size_t) key] == s) F->slot_of[(size_t) key] = -1;
                    ++held;
                } else if (st == FastState::kSpare) {
                    for (int j = 0; j < gf::kSpares; ++j)
                        if (PD.spare[j] == s) {
                            PD.spare[j] = -1;
                            ed.upd(F->spare_key(M.il, j), 0ull);
                        }
                } else if (st == FastState::kLanding || st == FastState::kDraining) {
                    err = "glm mtp lend: a draft-layer slot is still moving";
                    return false;
                }
                st = FastState::kLent;
                PD.key[(size_t) s] = -1;
            }
            if (!ed.flush()) return false;
        }
        // (3) the block's bytes into the pinned copy (MORE's weights have theirs from the start; its buffers need none)
        if (draft && (!glmfast::cuda_ok(cudaMemcpyAsync(M.snap, M.region, M.bytes, cudaMemcpyDeviceToHost, F->cs),
                                        "glm mtp lend copy out", err) ||
                      !glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm mtp lend copy out", err)))
            return false;
        // (4) the borrowed slots are the trunk's: free, for the boundary's spares and refill
        {
            std::lock_guard<std::mutex> lk(F->mu);
            for (int il = l0_; il < lt_; ++il) {
                auto& P = F->lp[(size_t) il];
                for (int s = P.n_tail; s < P.n; ++s)
                    if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
            }
            F->check_resident(l0_, lt_, NE);
        }
        F->bg_hold = false;
        M.lent = true;
        ++M.lends;
        const auto t1 = std::chrono::steady_clock::now();
        M.t_lent = t1;
        // the refill starts now (spares, then the hottest RAM-tier experts over the copy stream)
        if (!fast_boundary(err)) return false;
        const double all = secs(M.t_start, t1);
        size_t more_b = 0;
        for (const auto& r : M.more) more_b += r.bytes;
        char what[96];
        if (draft) std::snprintf(what, sizeof what, "the draft layer's %.2f GB (%d experts)", (double) M.bytes / 1073741824.0, held);
        else std::snprintf(what, sizeof what, "MTP's buffers (%.2f GB)", (double) more_b / 1073741824.0);
        std::fprintf(stderr, "glm mtp lend: CUDA%d lent %s to the trunk in %.0f ms: %d borrowed slots in %d layers%s for "
                             "the tier's refill - %s; lent %.0f%% of the time so far (%llu lends)\n",
                     dev_, what, 1000.0 * secs(t0, t1), M.slots, M.layers,
                     draft && more_b > 0 ? " (MTP's other memory too)" : "", why.c_str(),
                     100.0 * M.lent_s / std::max(1e-9, all), (unsigned long long) M.lends);
        return true;
    }
    // ---- reclaim: (1) + (2) the borrowed slots empty (moves retired first), by the prompt's lending rules
    const int64_t out0 = experts_out(F, l0_, l1_, M.il, NE);
    const uint64_t park0 = F->diag_lend_park;
    uint64_t moved = 0, evicted = 0;
    if (!lend_tail(0, moved, evicted, err, true)) return false;
    const uint64_t parked = F->diag_lend_park - park0;
    // (3) the draft layer's bytes back, behind those copies on the same stream; MORE's weights from their copies (its
    //     buffers are the windows' again as they are: each window writes them before it reads them)
    if (draft && !glmfast::cuda_ok(cudaMemcpyAsync(M.region, M.snap, M.bytes, cudaMemcpyHostToDevice, F->cs),
                                   "glm mtp lend copy in", err))
        return false;
    for (const auto& r : M.more)
        if (r.copy != nullptr &&
            !glmfast::cuda_ok(cudaMemcpyAsync(r.p, r.copy, r.bytes, cudaMemcpyHostToDevice, F->cs), "glm mtp lend copy in",
                              err))
            return false;
    // (4) its slot map and table entries as they were
    int back = 0, lost = 0;
    {
        std::lock_guard<std::mutex> lk(F->mu);
        TableEdits ed{F, err};
        for (int s = 0; s < PD.n_main; ++s) {
            const int e = M.key[(size_t) s];
            PD.tick[(size_t) s] = M.tick[(size_t) s];
            PD.st[(size_t) s] = FastState::kFree;
            PD.key[(size_t) s] = -1;
            if (e < 0) continue;
            const int key = M.il * NE + e;
            if (F->slot_of[(size_t) key] >= 0) {   // (it found another slot meanwhile - never expected: the boundary
                ++lost;                            //  leaves the draft layer alone while lent - that one stays)
                continue;
            }
            PD.st[(size_t) s] = FastState::kResident;
            PD.key[(size_t) s] = e;
            F->slot_of[(size_t) key] = s;
            ed.upd(F->tab_key(key), (unsigned long long) PD.slot_ptr(s));
            ++back;
        }
        if (!ed.flush()) return false;
        F->check_resident(l0_, lt_, NE);
    }
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm mtp lend copy in", err)) return false;
    // (what the parking could not hold - no free RAM slot, or a colder RAM expert it evicted - is in no tier now)
    const int64_t disk = std::max<int64_t>(0, experts_out(F, l0_, l1_, M.il, NE) - out0);
    M.lent = false;
    ++M.reclaims;
    const auto t1 = std::chrono::steady_clock::now();
    M.lent_s += secs(M.t_lent, t1);
    const double all = secs(M.t_start, t1);
    std::fprintf(stderr, "glm mtp lend: CUDA%d %s back in %.0f ms (%d experts restored%s): the borrowed slots' "
                         "%llu experts left them - %llu over colder main residents, %llu parked in the RAM tier, %lld "
                         "more in no tier (disk only) - %s; lent %.0f%% of the time so far (%llu reclaims)\n",
                 dev_, draft ? "the draft layer" : "MTP's buffers", 1000.0 * secs(t0, t1), back,
                 lost > 0 ? (", " + std::to_string(lost) + " found elsewhere").c_str() : "",
                 (unsigned long long) evicted, (unsigned long long) moved, (unsigned long long) parked,
                 (long long) disk, why.c_str(),
                 100.0 * M.lent_s / std::max(1e-9, all), (unsigned long long) M.reclaims);
    return true;
}

// (MAYA_GLM_MTP_LEND_MORE lends on every part: one that failed a switch stops the switching for all - nothing drafts)
bool Glm5Model::mtp_lend_ready() const {
    bool any = false;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || !m->fast_->ml.on) continue;
        if (m->fast_->ml.broken) return false;
        any = true;
    }
    return any;
}

bool Glm5Model::mtp_lent() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->fast_->ml.on && m->fast_->ml.lent) return true;
    return false;
}

bool Glm5Model::mtp_lend_switch(bool lend, const std::string& why, std::string& err) {
    bool ok = true;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {   // (every part that lends: MORE's too)
        if (m->fast_ == nullptr || !m->fast_->ml.on) continue;
        cudaSetDevice(m->dev_);
        if (!m->mtp_lend_part(lend, why, err)) ok = false;
    }
    cudaSetDevice(dev_);
    return ok;
}

}  // namespace strata::core
