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

#include "strata/core/mtp_lend.hpp"
#include "strata/core/tier_policy.hpp"

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
    const int d = mtp_il_;
    if (d < l0_ || d >= lt_ || !F->L[(size_t) d].moe) return true;   // (a part without the draft layer)
    const char* why = nullptr;
    if (multi_mtp_setting() <= 0) why = "it needs MAYA_GLM_MULTI_MTP";
    else if (n_parts_ < 2 || split_next_ != nullptr) why = "it needs a split (the draft layer on its last part)";
    else if (seq_count() < 2 || split_rows_setting() <= 0)
        why = "it needs --parallel and MAYA_GLM_SPLIT_ROWS (the windows' scheduler decides when)";
    else if (F->unified_memory || F->complete) why = "the pool holds every expert already";
    else if (F->lp[(size_t) d].n_main <= 0) why = "the draft layer has no main slots";
    if (why != nullptr) {
        std::fprintf(stderr, "glm mtp lend: CUDA%d off - %s\n", dev_, why);
        return true;
    }
    const auto& PD = F->lp[(size_t) d];
    M.il = d;
    M.region = PD.base;
    M.bytes = (size_t) PD.n_main * PD.stride;
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
    const std::vector<int> cnt = mtp_lend_counts(M.bytes, strides, have, g.n_expert + gf::kSpares, gain);
    const TailLayout lay = tier_tail_layout(strides, cnt, false);
    int tot = 0, layers = 0, cmin = INT_MAX, cmax = 0;
    for (size_t k = 0; k < ils.size(); ++k) {
        if (cnt[k] <= 0) continue;
        tot += cnt[k];
        ++layers;
        cmin = std::min(cmin, cnt[k]);
        cmax = std::max(cmax, cnt[k]);
    }
    if (tot == 0) {
        std::fprintf(stderr, "glm mtp lend: CUDA%d off - the draft layer's slots hold no trunk slot\n", dev_);
        return true;
    }
    // the pinned copy the draft layer's bytes wait in while lent
    if (cudaHostAlloc((void**) &M.snap, M.bytes, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        M.snap = nullptr;
        std::fprintf(stderr, "glm mtp lend: CUDA%d off - %.2f GB of pinned RAM for the draft layer's copy did not "
                             "allocate\n", dev_, (double) M.bytes / 1073741824.0);
        return true;
    }
    for (size_t k = 0; k < ils.size(); ++k) {
        if (cnt[k] <= 0) continue;
        auto& P = F->lp[(size_t) ils[k]];
        P.bbase = M.region + lay.off[k];
        P.n = P.n_tail + cnt[k];
        P.key.resize((size_t) P.n, -1);
        P.tick.resize((size_t) P.n, 0);
        P.st.resize((size_t) P.n, FastState::kLent);   // (the draft layer's memory until it is lent)
    }
    M.on = true;
    M.slots = tot;
    M.layers = layers;
    M.key.assign((size_t) PD.n_main, -1);
    M.tick.assign((size_t) PD.n_main, 0);
    const MtpLendPolicy& pol = mtp_lend_policy();
    std::fprintf(stderr, "glm mtp lend: CUDA%d the draft layer's %d main slots (%.2f GB) serve %d slots of %d trunk layers "
                         "(%d-%d a layer) while drafting is off (MAYA_GLM_MTP_LEND): lent after %.0f s with more than "
                         "MAYA_GLM_MULTI_MTP_MAX running, back after %.0f s at that or fewer (one or none: at once); "
                         "%.2f GB pinned for its copy\n",
                 dev_, PD.n_main, (double) M.bytes / 1073741824.0, tot, layers, cmin, cmax, pol.lend_after,
                 pol.reclaim_after, (double) M.bytes / 1073741824.0);
    return true;
}

void Glm5Model::mtp_lend_free() {
    FastState* F = fast_;
    if (F == nullptr) return;
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
    auto& PD = F->lp[(size_t) M.il];
    const auto t0 = std::chrono::steady_clock::now();
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
        // (3) the block's bytes into the pinned copy
        if (!glmfast::cuda_ok(cudaMemcpyAsync(M.snap, M.region, M.bytes, cudaMemcpyDeviceToHost, F->cs),
                              "glm mtp lend copy out", err) ||
            !glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm mtp lend copy out", err))
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
        std::fprintf(stderr, "glm mtp lend: CUDA%d lent the draft layer's %.2f GB (%d experts) to the trunk in %.0f ms: "
                             "%d borrowed slots in %d layers for the tier's refill - %s; lent %.0f%% of the time so far "
                             "(%llu lends)\n",
                     dev_, (double) M.bytes / 1073741824.0, held, 1000.0 * secs(t0, t1), M.slots, M.layers, why.c_str(),
                     100.0 * M.lent_s / std::max(1e-9, all), (unsigned long long) M.lends);
        return true;
    }
    // ---- reclaim: (1) + (2) the borrowed slots empty (moves retired first), by the prompt's lending rules
    const int64_t out0 = experts_out(F, l0_, l1_, M.il, NE);
    const uint64_t park0 = F->diag_lend_park;
    uint64_t moved = 0, evicted = 0;
    if (!lend_tail(0, moved, evicted, err, true)) return false;
    const uint64_t parked = F->diag_lend_park - park0;
    // (3) the draft layer's bytes back, behind those copies on the same stream
    if (!glmfast::cuda_ok(cudaMemcpyAsync(M.region, M.snap, M.bytes, cudaMemcpyHostToDevice, F->cs),
                          "glm mtp lend copy in", err))
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
    std::fprintf(stderr, "glm mtp lend: CUDA%d the draft layer back in %.0f ms (%d experts restored%s): the borrowed slots' "
                         "%llu experts left them - %llu over colder main residents, %llu parked in the RAM tier, %lld "
                         "more in no tier (disk only) - %s; lent %.0f%% of the time so far (%llu reclaims)\n",
                 dev_, 1000.0 * secs(t0, t1), back,
                 lost > 0 ? (", " + std::to_string(lost) + " found elsewhere").c_str() : "",
                 (unsigned long long) evicted, (unsigned long long) moved, (unsigned long long) parked,
                 (long long) disk, why.c_str(),
                 100.0 * M.lent_s / std::max(1e-9, all), (unsigned long long) M.reclaims);
    return true;
}

bool Glm5Model::mtp_lend_ready() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->fast_->ml.on && !m->fast_->ml.broken) return true;
    return false;
}

bool Glm5Model::mtp_lent() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->fast_->ml.on && m->fast_->ml.lent) return true;
    return false;
}

bool Glm5Model::mtp_lend_switch(bool lend, const std::string& why, std::string& err) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || !m->fast_->ml.on) continue;
        cudaSetDevice(m->dev_);
        const bool ok = m->mtp_lend_part(lend, why, err);
        cudaSetDevice(dev_);
        return ok;
    }
    return true;
}

}  // namespace strata::core
