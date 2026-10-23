#pragma once
// STRATA_GLM_MTP_LEND=1 (opt-in, needs STRATA_GLM_MULTI_MTP on a split): the NextN draft layer's expert slots - its main
// slots, one block of the expert pool on the split's last GPU - serve the trunk's experts while drafting is off, and
// come back for drafting.  The engine side is src/core/glm_mtp_lend.cu; here the pure rules, CPU-tested
// (tests/core/mtp_lend_test.cpp):
//
//   - WHEN (mtp_lend_step): the batched draft windows run only while STRATA_GLM_MULTI_MTP_MAX or fewer requests run, so
//     the slots are lent once more than that have run for STRATA_GLM_MTP_LEND_AFTER seconds (default 30) without a
//     break, and come back once that many or fewer have run for STRATA_GLM_MTP_RECLAIM_AFTER seconds (default 3) - at
//     once when one request runs (its speculative decode) or none.  Drafting is worth far more at MAX or fewer than the
//     slots are above it, and a switch costs a fraction of a second of the last GPU, hence the asymmetry: a count that
//     bounces around MAX never lends.  No switch while `blocked` (a prompt is being read): the timers run on, the switch
//     comes at the first step after.
//   - HOW MANY (mtp_lend_counts): the region's bytes as slots of the trunk layers, each next slot to the layer whose
//     next one serves the largest share of its routes (the pool's own split continued), never past a layer's cap.
//     Their places in the region: tier_tail_layout(strides, counts, false) (layer-major, each layer's own stride).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

namespace strata::core {

struct MtpLendPolicy {
    bool on = false;
    double lend_after = 30.0;     // seconds with more than `max` running before the slots are lent
    double reclaim_after = 3.0;   // seconds with `max` or fewer running before they come back (1 or 0 running: at once)
};

/// The settings' texts (null: unset) -> the policy.  Out-of-range values keep the default.
inline MtpLendPolicy mtp_lend_parse(const char* on, const char* lend_after, const char* reclaim_after) {
    MtpLendPolicy p;
    p.on = on != nullptr && std::atoi(on) != 0;
    if (lend_after != nullptr && lend_after[0] != '\0') {
        const double v = std::atof(lend_after);
        if (v >= 0.0 && v <= 86400.0) p.lend_after = v;
    }
    if (reclaim_after != nullptr && reclaim_after[0] != '\0') {
        const double v = std::atof(reclaim_after);
        if (v >= 0.0 && v <= 86400.0) p.reclaim_after = v;
    }
    return p;
}

/// The engine's policy, read once.
inline const MtpLendPolicy& mtp_lend_policy() {
    static const MtpLendPolicy p = mtp_lend_parse(std::getenv("STRATA_GLM_MTP_LEND"),
                                                  std::getenv("STRATA_GLM_MTP_LEND_AFTER"),
                                                  std::getenv("STRATA_GLM_MTP_RECLAIM_AFTER"));
    return p;
}

/// What else lending does (each opt-in, each needs STRATA_GLM_MTP_LEND):
///   - skip_read (STRATA_GLM_MTP_LEND_SKIP_READ=1): a prompt read while the slots are lent leaves the NextN block's
///     cache fill out (its eh_proj, norms and cache projections, and the host's dequantized next-token rows); the
///     trunk's hidden states of the last `catchup` positions a conversation's reads left out are kept (pinned RAM,
///     4 n_embd bytes a position) and fill its NextN cache before it drafts again (STRATA_GLM_MTP_LEND_CATCHUP,
///     default 2048; 0: none - those entries stay as they were, which only costs draft acceptance).
///   - more (STRATA_GLM_MTP_LEND_MORE=1): MTP's other VRAM is lent too - every part's verify-window buffers (the
///     conversations' recurrent-state copies, the windows' KDA inputs, their residual and draft rows: one block a part
///     then) and, with skip_read, the NextN block's dense weights and a device token_embd - as more borrowed slots,
///     the first GPUs' trunk layers included.  Weights come back from a pinned copy taken at the start, the buffers are
///     simply theirs again (the next window writes them before reading).
struct MtpLendExtra {
    bool skip_read = false;
    int catchup = 2048;   // positions a conversation (with skip_read)
    bool more = false;
};

inline MtpLendExtra mtp_lend_extra_parse(const char* skip_read, const char* catchup, const char* more = nullptr) {
    MtpLendExtra x;
    x.more = more != nullptr && std::atoi(more) != 0;
    x.skip_read = skip_read != nullptr && std::atoi(skip_read) != 0;
    if (catchup != nullptr && catchup[0] != '\0') {
        const long v = std::atol(catchup);
        if (v >= 0 && v <= (1L << 20)) x.catchup = (int) v;
    }
    return x;
}

inline const MtpLendExtra& mtp_lend_extra() {
    static const MtpLendExtra x =
        mtp_lend_extra_parse(std::getenv("STRATA_GLM_MTP_LEND_SKIP_READ"), std::getenv("STRATA_GLM_MTP_LEND_CATCHUP"),
                             std::getenv("STRATA_GLM_MTP_LEND_MORE"));
    return x;
}

/// One conversation's catch-up entries (skip_read): the positions whose NextN cache entry a read left out, the newest
/// `cap` of them, in a ring whose index is also the row of the trunk's hidden state the engine keeps for it.  Entries
/// go in by position (a read's chunks in order); a read at p first drops every entry at p or after (it rewrites them),
/// so the ring stays sorted oldest -> newest.
struct MtpCatchRing {
    int cap = 0, head = 0, count = 0;   // head: the next index written; the newest `count` entries end before it
    std::vector<int64_t> pos;
    std::vector<int32_t> next;   // the token at pos + 1 (the entry's input beside the hidden state)
    uint64_t lost = 0;           // entries pushed out by newer ones before a catch-up took them

    void init(int c) {
        cap = c > 0 ? c : 0;
        head = count = 0;
        pos.assign((size_t) cap, -1);
        next.assign((size_t) cap, -1);
        lost = 0;
    }
    int at(int k) const { return (head - count + k + 2 * cap) % cap; }   // the k-th oldest entry's index
    /// every entry at position p or after goes (they are the newest: the ring is sorted)
    void drop_from(int64_t p) {
        while (count > 0 && pos[(size_t) at(count - 1)] >= p) {
            --count;
            head = (head - 1 + cap) % cap;
        }
    }
    void clear() { head = count = 0; }
    /// n entries at positions p0 .. p0 + n - 1 (tokens nx[0..n-1]); only the newest min(n, cap) are kept.  The rows go
    /// to the indices returned as up to two runs: seg[i] = {first index, rows, the run's first row in the input}.
    /// Returns the number of runs.
    int append(int64_t p0, int n, const int32_t* nx, int seg[2][3]) {
        if (cap <= 0 || n <= 0) return 0;
        const int skip = n > cap ? n - cap : 0;
        const int keep = n - skip;
        lost += (uint64_t) std::max(0, count + keep - cap) + (uint64_t) skip;
        int ns = 0, done = 0;
        while (done < keep) {
            const int run = std::min(keep - done, cap - head);
            seg[ns][0] = head;
            seg[ns][1] = run;
            seg[ns][2] = skip + done;
            ++ns;
            for (int i = 0; i < run; ++i) {
                pos[(size_t) head + (size_t) i] = p0 + skip + done + i;
                next[(size_t) head + (size_t) i] = nx[skip + done + i];
            }
            head = (head + run) % cap;
            done += run;
        }
        count = std::min(cap, count + keep);
        return ns;
    }
    /// The entries a catch-up fills, oldest first, as runs of consecutive positions at consecutive indices, at most
    /// `rows` each; only entries whose next token is behind `pend` (the conversation's position now) count.
    /// fn(index, position, rows) per run.
    template <class Fn> void runs(int64_t pend, int rows, Fn&& fn) const {
        int k = 0;
        while (k < count) {
            const int i0 = at(k);
            const int64_t p = pos[(size_t) i0];
            if (p < 0 || p + 1 >= pend) {
                ++k;
                continue;
            }
            int n = 1;
            while (k + n < count && n < rows) {
                const int i = at(k + n);
                if (i != i0 + n || pos[(size_t) i] != p + n || p + n + 1 >= pend) break;
                ++n;
            }
            fn(i0, p, n);
            k += n;
        }
    }
};

enum class MtpLendAct { none, lend, reclaim };

struct MtpLendClock {
    bool lent = false;     // the slots serve the trunk now
    double since = -1.0;   // when the count last crossed to the side that wants the other state (-1: it does not)
};

/// One scheduler step at time `now` (seconds) with `running` requests (decoding or waiting for a context) and the
/// windows' MAX (<= 0: windows always - nothing is ever lent).  What to do now; mtp_lend_done records that it was done.
inline MtpLendAct mtp_lend_step(MtpLendClock& c, const MtpLendPolicy& p, double now, int running, int max,
                                bool blocked) {
    if (!p.on) return MtpLendAct::none;
    const bool over = max > 0 && running > max;
    if (!c.lent) {
        if (!over) {
            c.since = -1.0;
            return MtpLendAct::none;
        }
        if (c.since < 0.0) c.since = now;
        return !blocked && now - c.since >= p.lend_after ? MtpLendAct::lend : MtpLendAct::none;
    }
    if (over) {
        c.since = -1.0;
        return MtpLendAct::none;
    }
    if (c.since < 0.0) c.since = now;
    return !blocked && (running <= 1 || now - c.since >= p.reclaim_after) ? MtpLendAct::reclaim : MtpLendAct::none;
}

inline void mtp_lend_done(MtpLendClock& c, bool lent) {
    c.lent = lent;
    c.since = -1.0;
}

/// The borrowed slots per layer: `region` bytes, layer k's slot `stride[k]` bytes, `have[k]` slots already (main +
/// tail), at most `cap` in all; gain(k, j): the share of layer k's routes its j-th slot serves (0 where unknown - the
/// layer with the fewest slots then goes first, lowest k on a tie).  A layer whose stride is 0 takes none.
inline std::vector<int> mtp_lend_counts(size_t region, const std::vector<size_t>& stride, const std::vector<int>& have,
                                        int cap, const std::function<double(int, int)>& gain) {
    const size_t n = stride.size();
    std::vector<int> cnt(n, 0);
    size_t left = region;
    for (;;) {
        int best = -1;
        double bg = -1.0;
        int bh = 0;
        for (size_t k = 0; k < n; ++k) {
            if (stride[k] == 0 || stride[k] > left) continue;
            const int h = have[k] + cnt[k];
            if (h >= cap) continue;
            const double gv = gain ? gain((int) k, h) : 0.0;
            if (best < 0 || gv > bg || (gv == bg && h < bh)) {
                best = (int) k;
                bg = gv;
                bh = h;
            }
        }
        if (best < 0) break;
        ++cnt[(size_t) best];
        left -= stride[(size_t) best];
    }
    return cnt;
}

/// The borrowed slots over SEVERAL regions (STRATA_GLM_MTP_LEND_MORE: the draft layer's slots, the windows' buffers, the
/// NextN block's weights): the same greedy as mtp_lend_counts - each next slot to the layer whose next one serves the
/// largest share, among the layers whose slot still fits some region - each slot at the front of the first region it
/// fits.  Per layer, its slots' {region, offset} in the order given; the slots tile each region's front, no overlap.
struct MtpLendSlot {
    int region = 0;
    size_t off = 0;
};
inline std::vector<std::vector<MtpLendSlot>> mtp_lend_place(const std::vector<size_t>& region,
                                                            const std::vector<size_t>& stride,
                                                            const std::vector<int>& have, int cap,
                                                            const std::function<double(int, int)>& gain) {
    const size_t n = stride.size();
    std::vector<std::vector<MtpLendSlot>> out(n);
    std::vector<size_t> left = region;
    const auto fit = [&](size_t b) -> int {
        for (size_t r = 0; r < left.size(); ++r)
            if (left[r] >= b) return (int) r;
        return -1;
    };
    for (;;) {
        int best = -1, bh = 0;
        double bg = -1.0;
        for (size_t k = 0; k < n; ++k) {
            if (stride[k] == 0 || fit(stride[k]) < 0) continue;
            const int h = have[k] + (int) out[k].size();
            if (h >= cap) continue;
            const double gv = gain ? gain((int) k, h) : 0.0;
            if (best < 0 || gv > bg || (gv == bg && h < bh)) {
                best = (int) k;
                bg = gv;
                bh = h;
            }
        }
        if (best < 0) break;
        const int r = fit(stride[(size_t) best]);
        out[(size_t) best].push_back({r, region[(size_t) r] - left[(size_t) r]});
        left[(size_t) r] -= stride[(size_t) best];
    }
    return out;
}

}  // namespace strata::core
