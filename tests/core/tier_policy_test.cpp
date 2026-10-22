// tier_policy (STRATA_GLM_COUNT_HALF, STRATA_GLM_PROMPT_COUNTS, STRATA_GLM_PROMOTE_SWAP): the parse, the defaults
// that keep the rule as before, the prompt's weighted counts, the aging both paths share, and the swap test.  CPU only.
#include "strata/core/tier_policy.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}
}  // namespace

int main() {
    using namespace strata::core;
    // unset: the rule as before
    {
        const TierPolicy p = tier_policy_parse(nullptr, nullptr, nullptr);
        expect(p.half == 32768 && p.prompt_w == 1.0 && !p.swap_any, "defaults");
        const TierPolicy q = tier_policy_parse("", "", "");
        expect(q.half == 32768 && q.prompt_w == 1.0 && !q.swap_any, "empty texts are unset");
    }
    // the parse and its ranges
    {
        const TierPolicy p = tier_policy_parse("262144", "0.25", "1");
        expect(p.half == 262144 && p.prompt_w == 0.25 && p.swap_any, "values");
        expect(tier_policy_parse("1e6", nullptr, nullptr).half == 1000000, "an exponent");
        expect(tier_policy_parse("100", nullptr, nullptr).half == 32768, "too short a period keeps the default");
        expect(tier_policy_parse(nullptr, "1.5", nullptr).prompt_w == 1.0, "a weight above 1 keeps the default");
        expect(tier_policy_parse(nullptr, "-1", nullptr).prompt_w == 1.0, "a negative weight keeps the default");
        expect(tier_policy_parse(nullptr, "0", nullptr).prompt_w == 0.0, "weight 0");
        expect(!tier_policy_parse(nullptr, nullptr, "0").swap_any, "swap 0");
    }
    // the prompt's counts
    {
        expect(tier_prompt_count(37, 1.0) == 37, "weight 1 is the count");
        expect(tier_prompt_count(37, 0.0) == 0, "weight 0 adds nothing");
        expect(tier_prompt_count(37, 0.1) == 4, "weight 0.1 rounds");
        expect(tier_prompt_count(4, 0.1) == 0, "a small count at a small weight is 0");
    }
    // the aging: the same halvings as the engine's old loops at the default period
    {
        // the prompt's old loop: events += ev; while (events >= 32768) { events -= 32768; halve all }
        for (uint64_t start : {0ull, 100ull, 32760ull}) {
            for (uint64_t add : {0ull, 8ull, 4728ull, 32768ull, 99300ull, 1333000ull}) {
                uint64_t old_ev = start + add;
                int old_n = 0;
                while (old_ev >= 32768) {
                    old_ev -= 32768;
                    ++old_n;
                }
                uint64_t ev = start;
                const int n = tier_halvings(ev, add, 32768);
                expect(n == old_n && ev == old_ev, "halvings match the old loop");
            }
        }
        uint32_t c = 1000;
        for (int k = 0; k < 3; ++k) c >>= 1;
        expect(tier_halved(1000, 3) == c, "n halvings at once");
        expect(tier_halved(0xFFFFFFFFu, 32) == 0 && tier_halved(5, 40) == 0, "32+ halvings are 0");
        expect(tier_halved(7, 0) == 7, "no halving");
        uint64_t ev = 0;
        expect(tier_halvings(ev, 1000000, 262144) == 3 && ev == 1000000 - 3 * 262144, "a longer period");
        uint64_t z = 5;
        expect(tier_halvings(z, 100, 0) == 0, "period 0 never halves");
        // decode: 8 events a route; one halving per 4096 routes at the default
        uint64_t d = 0;
        int nh = 0;
        for (int r = 0; r < 4096 * 3; ++r) nh += tier_halvings(d, 8, 32768);
        expect(nh == 3 && d == 0, "4096 routes a halving");
    }
    // the swap test (fast_boundary's: hot >= ratio * max(cold, 1) + 4) and when swaps run
    {
        expect(tier_swap_due(6, 0, 2) && !tier_swap_due(5, 0, 2), "cold 0 counts as 1");
        expect(tier_swap_due(24, 10, 2) && !tier_swap_due(23, 10, 2), "ratio 2 plus 4");
        expect(!tier_swap_due(0, 0, 2), "nothing hot");
        expect(tier_swap_due(1u << 24, (1u << 23) - 2u, 2) && !tier_swap_due(1u << 24, (1u << 23) - 1u, 2),
               "at the counts' cap (1 << 24)");
        TierPolicy p;
        expect(tier_swaps_on(true, p) && !tier_swaps_on(false, p), "default: all-CPU plan only");
        p.swap_any = true;
        expect(tier_swaps_on(false, p), "swap_any: any plan");
    }
    if (fails == 0) std::printf("tier_policy_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
