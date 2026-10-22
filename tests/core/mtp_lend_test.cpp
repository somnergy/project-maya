// mtp_lend (STRATA_GLM_MTP_LEND, _LEND_AFTER, _RECLAIM_AFTER): the parse, the lend / reclaim state machine and its
// hysteresis (a count that bounces around MAX never lends; a dip resets the clock; a lone request takes the slots back
// at once; no switch while blocked, the switch at the first step after), and the borrowed slots per layer (greedy by
// share, capped, inside the region) laid out with tier_tail_layout (tiles at most the region, no overlap).  CPU only.
#include "strata/core/mtp_lend.hpp"
#include "strata/core/tier_policy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}

using strata::core::MtpLendAct;
using strata::core::MtpLendClock;
using strata::core::MtpLendPolicy;

// a run of the scheduler: (time, running) samples every `dt` seconds; each act is done at once (the engine's switch);
// returns the acts in order with their times
struct Ev {
    double t;
    MtpLendAct a;
};
std::vector<Ev> drive(MtpLendClock& c, const MtpLendPolicy& p, int max, const std::vector<std::pair<double, int>>& seg,
                      double dt, bool (*blocked)(double) = nullptr) {
    std::vector<Ev> out;
    double t = 0.0;
    for (const auto& s : seg) {
        for (double end = t + s.first; t < end - 1e-9; t += dt) {
            const MtpLendAct a = strata::core::mtp_lend_step(c, p, t, s.second, max, blocked ? blocked(t) : false);
            if (a != MtpLendAct::none) {
                out.push_back({t, a});
                strata::core::mtp_lend_done(c, a == MtpLendAct::lend);
            }
        }
    }
    return out;
}
}  // namespace

int main() {
    using namespace strata::core;
    // the parse
    {
        const MtpLendPolicy d = mtp_lend_parse(nullptr, nullptr, nullptr);
        expect(!d.on && d.lend_after == 30.0 && d.reclaim_after == 3.0, "defaults: off, 30 s, 3 s");
        const MtpLendPolicy p = mtp_lend_parse("1", "12.5", "0");
        expect(p.on && p.lend_after == 12.5 && p.reclaim_after == 0.0, "values");
        expect(!mtp_lend_parse("0", nullptr, nullptr).on, "0 is off");
        expect(mtp_lend_parse("1", "-3", "")  .lend_after == 30.0, "a negative time keeps the default");
        expect(mtp_lend_parse("1", "", "1e9").reclaim_after == 3.0, "too long a time keeps the default");
    }
    MtpLendPolicy p;
    p.on = true;
    p.lend_after = 30.0;
    p.reclaim_after = 3.0;
    // off: never anything
    {
        MtpLendClock c;
        MtpLendPolicy off = p;
        off.on = false;
        expect(drive(c, off, 4, {{100.0, 8}}, 0.5).empty(), "off: never lends");
    }
    // above MAX for long enough: lent once, at LEND_AFTER
    {
        MtpLendClock c;
        const auto ev = drive(c, p, 4, {{10.0, 3}, {100.0, 6}}, 0.25);
        expect(ev.size() == 1 && ev[0].a == MtpLendAct::lend, "sustained load: one lend");
        expect(!ev.empty() && ev[0].t >= 40.0 - 1e-9 && ev[0].t < 40.5, "the lend comes LEND_AFTER after the count passed MAX");
        expect(c.lent, "lent after it");
    }
    // exactly MAX is not above it
    {
        MtpLendClock c;
        expect(drive(c, p, 4, {{200.0, 4}}, 0.5).empty(), "MAX running: never lends");
    }
    // a dip to MAX or fewer resets the clock: a bounce 4 <-> 5 every few seconds never lends
    {
        MtpLendClock c;
        std::vector<std::pair<double, int>> seg;
        for (int i = 0; i < 100; ++i) {
            seg.push_back({7.0, 5});
            seg.push_back({2.0, 4});
        }
        expect(drive(c, p, 4, seg, 0.25).empty(), "a bouncing count never lends");
        MtpLendClock c2;
        const auto ev = drive(c2, p, 4, {{29.0, 6}, {0.5, 3}, {29.0, 6}, {5.0, 6}}, 0.25);
        expect(ev.size() == 1 && ev[0].t > 59.0, "a dip restarts the clock");
    }
    // reclaim: RECLAIM_AFTER at 2..MAX; at once for one request or none; a count above MAX again resets it
    {
        MtpLendClock c;
        auto ev = drive(c, p, 4, {{40.0, 8}, {10.0, 3}}, 0.25);
        expect(ev.size() == 2 && ev[1].a == MtpLendAct::reclaim, "load falls to 3: reclaimed");
        expect(ev.size() == 2 && ev[1].t >= 43.0 - 1e-9 && ev[1].t < 43.5, "after RECLAIM_AFTER");
        expect(!c.lent, "owned after it");
        MtpLendClock c2;
        ev = drive(c2, p, 4, {{40.0, 8}, {1.0, 1}}, 0.25);
        expect(ev.size() == 2 && ev[1].a == MtpLendAct::reclaim && ev[1].t < 40.0 + 1e-9, "a lone request: at once");
        MtpLendClock c3;
        ev = drive(c3, p, 4, {{40.0, 8}, {0.5, 0}}, 0.25);
        expect(ev.size() == 2 && ev[1].a == MtpLendAct::reclaim && ev[1].t < 40.0 + 1e-9, "idle: at once");
        MtpLendClock c4;
        ev = drive(c4, p, 4, {{40.0, 8}, {2.0, 3}, {1.0, 6}, {2.0, 3}, {1.0, 6}}, 0.25);
        expect(ev.size() == 1, "dips shorter than RECLAIM_AFTER keep it lent");
    }
    // MAX <= 0 (windows always): nothing to lend
    {
        MtpLendClock c;
        expect(drive(c, p, 0, {{200.0, 8}}, 0.5).empty(), "MAX 0: never lends");
    }
    // blocked (a prompt being read): no switch, the clock runs on, the switch at the first step after
    {
        MtpLendClock c;
        const auto ev = drive(c, p, 4, {{60.0, 8}}, 0.25, [](double t) { return t < 50.0; });
        expect(ev.size() == 1 && ev[0].a == MtpLendAct::lend && ev[0].t >= 50.0 - 1e-9 && ev[0].t < 50.3,
               "blocked: the lend waits for the first free step");
        MtpLendClock c2;
        const auto ev2 = drive(c2, p, 4, {{40.0, 8}, {10.0, 1}}, 0.25, [](double t) { return t >= 40.0 && t < 45.0; });
        expect(ev2.size() == 2 && ev2[1].t >= 45.0 - 1e-9 && ev2[1].t < 45.3, "blocked: the reclaim waits too");
    }
    // the switch rate is bounded by LEND_AFTER whatever the pattern (8 for 31 s, then 1, repeated)
    {
        MtpLendClock c;
        std::vector<std::pair<double, int>> seg;
        for (int i = 0; i < 20; ++i) {
            seg.push_back({31.0, 8});
            seg.push_back({1.0, 1});
        }
        const auto ev = drive(c, p, 4, seg, 0.25);
        int lends = 0;
        for (const auto& e : ev) lends += e.a == MtpLendAct::lend;
        expect(lends == 20, "one lend per cycle of 32 s");
        bool alt = true;
        for (size_t i = 0; i < ev.size(); ++i) alt = alt && ev[i].a == (i % 2 == 0 ? MtpLendAct::lend : MtpLendAct::reclaim);
        expect(alt, "lend and reclaim alternate");
    }
    // the borrowed slots: Maya-L's CUDA1 - 21 trunk layers of 11.92 MB, a region of 222 draft slots of 14.93 MB
    {
        const size_t trunk = 11920000, draft = 14930000;
        const size_t region = 222 * draft;
        std::vector<size_t> st(21, trunk);
        std::vector<int> have(21, 0);
        for (int k = 0; k < 21; ++k) have[(size_t) k] = 204 + (k * 47) % 48;   // 204..251, as the start log's
        // shares: a layer's j-th slot serves 1/(1+j) of a per-layer weight (early layers spread wider)
        const auto gain = [&](int k, int j) { return (1.0 + 0.05 * (21 - k)) / (1.0 + j); };
        const std::vector<int> cnt = mtp_lend_counts(region, st, have, 288 + 3, gain);
        size_t used = 0;
        int tot = 0;
        for (size_t k = 0; k < cnt.size(); ++k) {
            used += (size_t) cnt[k] * st[k];
            tot += cnt[k];
        }
        expect(used <= region && region - used < trunk, "the counts fill the region to less than a slot");
        expect(tot == (int) (region / trunk), "as many trunk slots as the region holds");
        bool capped = true;
        for (size_t k = 0; k < cnt.size(); ++k) capped = capped && have[k] + cnt[k] <= 291;
        expect(capped, "no layer past its cap");
        // greedy by share: the last slot given anywhere is worth no less than any slot not given
        double worst_given = 1e9, best_not = 0.0;
        for (size_t k = 0; k < cnt.size(); ++k) {
            if (cnt[k] > 0) worst_given = std::min(worst_given, gain((int) k, have[k] + cnt[k] - 1));
            if (have[k] + cnt[k] < 291) best_not = std::max(best_not, gain((int) k, have[k] + cnt[k]));
        }
        expect(worst_given >= best_not, "greedy: no unchosen slot beats a chosen one");
        // laid out layer-major in the region: tiles [0, used), no overlap, inside the region
        const TailLayout L = tier_tail_layout(st, cnt, false);
        std::vector<std::pair<size_t, size_t>> iv;
        for (size_t k = 0; k < st.size(); ++k)
            for (int t = 0; t < cnt[k]; ++t) iv.push_back({L.off[k] + (size_t) t * st[k], L.off[k] + (size_t) (t + 1) * st[k]});
        std::sort(iv.begin(), iv.end());
        bool tiled = !iv.empty() && iv.front().first == 0 && iv.back().second == used;
        for (size_t i = 1; i < iv.size(); ++i) tiled = tiled && iv[i].first == iv[i - 1].second;
        expect(tiled && used <= region, "layout: the borrowed slots tile the region's front");
    }
    // caps, strides of 0, mixed strides, no shares (round-robin by fewest slots)
    {
        const std::vector<int> c1 =
            mtp_lend_counts(1000, {100, 100}, {290, 10}, 291, [](int k, int) { return k == 0 ? 1.0 : 0.5; });
        expect(c1[0] == 1 && c1[1] == 9, "a layer at its cap takes no more");
        const std::vector<int> c2 = mtp_lend_counts(1000, {0, 100}, {0, 0}, 291, nullptr);
        expect(c2[0] == 0 && c2[1] == 10, "a stride of 0 takes none");
        const std::vector<int> c3 = mtp_lend_counts(600, {100, 100, 100}, {5, 5, 5}, 291, nullptr);
        expect(c3[0] == 2 && c3[1] == 2 && c3[2] == 2, "no shares: round-robin by fewest slots");
        const std::vector<int> c4 = mtp_lend_counts(350, {200, 100}, {0, 0}, 291, nullptr);
        expect(c4[0] * 200 + c4[1] * 100 <= 350 && c4[0] + c4[1] >= 2, "mixed strides fit the region");
        const std::vector<int> c5 = mtp_lend_counts(50, {100}, {0}, 291, nullptr);
        expect(c5[0] == 0, "a region smaller than a slot lends nothing");
    }
    if (fails == 0) std::printf("mtp_lend_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
