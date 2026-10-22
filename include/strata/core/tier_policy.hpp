#pragma once
// The decode expert tier's aged counts and background swaps, opt-in (each unset = the rule as before):
//
//   STRATA_GLM_COUNT_HALF=<events>   the host's aged counts (what a boundary evicts, refills and swaps by) are halved
//                                    every this many count events, 8 a decode route (default 32768: 4096 routes of a
//                                    part, a few seconds of a batched decode).  Several conversations take turns and
//                                    pause for their tools; a longer memory keeps the experts all of them use.
//   STRATA_GLM_PROMPT_COUNTS=<w>     a prompt chunk's routing enters those counts with weight w (default 1).  A chunk
//                                    adds T x 8 events a layer - one read of a few hundred tokens halves every count
//                                    several times, a long one wipes them - so after a read the evictions follow that
//                                    one prompt.  0 = prompts neither add to the counts nor age them.
//   STRATA_GLM_PROMOTE_SWAP=1        the background swaps (a layer's coldest resident demoted when a RAM-only expert is
//                                    used bg_ratio times as much, the hot one promoted into the slot at a later
//                                    boundary; STRATA_GLM_PROMOTE moves a boundary) run with any CPU-lane plan, not
//                                    only the all-CPU one.  With a mixed plan the lane takes a route's lone miss, so an
//                                    expert that keeps missing alone is computed on the CPU every time and never
//                                    fetched - never promoted by a route.
//
// Pure functions, CPU-tested (tests/core/tier_policy_test.cpp).
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace strata::core {

struct TierPolicy {
    uint64_t half = 32768;    // count events between two halvings of the host's aged counts
    double prompt_w = 1.0;    // a prompt route's weight in those counts (and in their aging)
    bool swap_any = false;    // background swaps with any CPU-lane plan
};

/// The three settings' texts (null: unset) -> the policy.  Out-of-range values keep the default.
inline TierPolicy tier_policy_parse(const char* half, const char* prompt_w, const char* swap) {
    TierPolicy p;
    if (half != nullptr && half[0] != '\0') {
        const double v = std::atof(half);
        if (v >= 1024.0 && v <= 1e12) p.half = (uint64_t) v;
    }
    if (prompt_w != nullptr && prompt_w[0] != '\0') {
        const double v = std::atof(prompt_w);
        if (v >= 0.0 && v <= 1.0) p.prompt_w = v;
    }
    if (swap != nullptr && swap[0] != '\0') p.swap_any = std::atoi(swap) != 0;
    return p;
}

/// The engine's policy, read once.
inline const TierPolicy& tier_policy() {
    static const TierPolicy p = tier_policy_parse(std::getenv("STRATA_GLM_COUNT_HALF"),
                                                  std::getenv("STRATA_GLM_PROMPT_COUNTS"),
                                                  std::getenv("STRATA_GLM_PROMOTE_SWAP"));
    return p;
}

/// A prompt chunk's count of an expert as it enters the aged counts (rounded to nearest; 0 when the weight is 0).
inline uint64_t tier_prompt_count(uint64_t cnt, double w) {
    if (w >= 1.0) return cnt;
    if (w <= 0.0) return 0;
    return (uint64_t) std::llround((double) cnt * w);
}

/// `add` more count events: how many halvings of every count are due now (events keeps the remainder).
/// The decode's own aging (one route = 8 events) and the prompt's take this one rule.
inline int tier_halvings(uint64_t& events, uint64_t add, uint64_t half) {
    if (half == 0) return 0;
    events += add;
    int n = 0;
    while (events >= half) {
        events -= half;
        ++n;
    }
    return n;
}

/// One count after n halvings (a count that halved 32 times or more is 0).
inline uint32_t tier_halved(uint32_t c, int n) { return n >= 32 ? 0u : c >> n; }

/// A background swap's test: the hot RAM-only expert's count against the layer's coldest resident's.
inline bool tier_swap_due(uint32_t hot, uint32_t cold, uint32_t ratio) {
    return (uint64_t) hot >= (uint64_t) ratio * (cold > 1u ? cold : 1u) + 4u;
}

/// Whether the background swaps run: the all-CPU plan (as before), or any plan with STRATA_GLM_PROMOTE_SWAP.
inline bool tier_swaps_on(bool lane_all, const TierPolicy& p) { return lane_all || p.swap_any; }

}  // namespace strata::core
