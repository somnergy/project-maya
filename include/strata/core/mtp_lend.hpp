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

}  // namespace strata::core
