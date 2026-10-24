// mtp_lend (STRATA_GLM_MTP_LEND, _LEND_AFTER, _RECLAIM_AFTER): the parse, the lend / reclaim state machine and its
// hysteresis (a count that bounces around MAX never lends; a dip resets the clock; a lone request takes the slots back
// at once; no switch while blocked, the switch at the first step after), and the borrowed slots per layer (greedy by
// share, capped, inside the region) laid out with tier_tail_layout (tiles at most the region, no overlap).  CPU only.
#include "strata/core/mtp_lend.hpp"
#include "strata/core/tier_policy.hpp"

#include <algorithm>
#include <array>
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
using strata::core::MtpLendExtra;
using strata::core::MtpCatchRing;

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
    // STRATA_GLM_MTP_LEND_SKIP_READ / _CATCHUP: the parse
    {
        const MtpLendExtra d = mtp_lend_extra_parse(nullptr, nullptr);
        expect(!d.skip_read && d.catchup == 2048, "extra defaults: no skip, 2048 positions");
        const MtpLendExtra x = mtp_lend_extra_parse("1", "0");
        expect(x.skip_read && x.catchup == 0, "skip without a catch-up");
        expect(mtp_lend_extra_parse("1", "-5").catchup == 2048, "a negative catch-up keeps the default");
        expect(mtp_lend_extra_parse("1", "4096").catchup == 4096, "catch-up 4096");
    }
    // the catch-up ring: a read's chunks in order, the newest cap kept, a rewind drops the suffix, runs for the fill
    {
        MtpCatchRing r;
        r.init(8);
        std::vector<int32_t> nx(20);
        for (int i = 0; i < 20; ++i) nx[(size_t) i] = 1000 + i;
        int seg[2][3];
        // 5 rows at 100..104
        int ns = r.append(100, 5, nx.data(), seg);
        expect(ns == 1 && seg[0][0] == 0 && seg[0][1] == 5 && seg[0][2] == 0 && r.count == 5, "append 5 into 8");
        // 6 more at 105..110: 3 fit before the end, 3 wrap; 3 old ones pushed out
        ns = r.append(105, 6, nx.data(), seg);
        expect(ns == 2 && seg[0][0] == 5 && seg[0][1] == 3 && seg[0][2] == 0 && seg[1][0] == 0 && seg[1][1] == 3 &&
                   seg[1][2] == 3,
               "a wrap: two runs");
        expect(r.count == 8 && r.lost == 3, "the ring keeps the newest 8, 3 lost");
        expect(r.pos[(size_t) r.at(0)] == 103 && r.pos[(size_t) r.at(7)] == 110, "oldest 103, newest 110");
        expect(r.next[(size_t) r.at(0)] == 1003 && r.next[(size_t) r.at(7)] == 1005, "tokens by row");
        // runs for a conversation at position 200: oldest first, split where the indices wrap, at most 4 rows each
        std::vector<std::array<int64_t, 3>> got;
        r.runs(200, 4, [&](int i, int64_t p, int n) { got.push_back({(int64_t) i, p, (int64_t) n}); });
        bool ok = got.size() == 3 && got[0][1] == 103 && got[0][2] == 4 && got[1][1] == 107 && got[1][2] == 1 &&
                  got[2][1] == 108 && got[2][2] == 3 && got[2][0] == 0;
        int64_t sum = 0;
        for (const auto& g : got) sum += g[2];
        expect(ok && sum == 8, "runs: 103-106 (4 rows at most), 107 (the ring's end), 108-110");
        // only entries whose next token is behind the conversation's position
        got.clear();
        r.runs(108, 16, [&](int i, int64_t p, int n) { got.push_back({(int64_t) i, p, (int64_t) n}); });
        sum = 0;
        for (const auto& g : got) sum += g[2];
        expect(sum == 4 && got.back()[1] + got.back()[2] - 1 == 106, "entries up to 106 at position 108");
        // a read at 107 rewrites 107.. : they go, 103..106 stay
        r.drop_from(107);
        expect(r.count == 4 && r.pos[(size_t) r.at(3)] == 106, "a rewind drops the newest");
        ns = r.append(107, 2, nx.data(), seg);
        expect(ns == 2 && r.count == 6 && r.pos[(size_t) r.at(5)] == 108 && r.next[(size_t) r.at(4)] == 1000,
               "appended after the drop (across the ring's end)");
        // a gap between reads (decode positions in between): two runs
        ns = r.append(150, 2, nx.data() + 10, seg);
        got.clear();
        r.runs(1000, 16, [&](int i, int64_t p, int n) { got.push_back({(int64_t) i, p, (int64_t) n}); });
        expect(r.count == 8 && got.size() >= 2 && got.back()[1] == 150 && got.back()[2] == 2, "a gap splits the runs");
        // more rows than the ring: the newest cap of them
        MtpCatchRing q;
        q.init(4);
        ns = q.append(0, 10, nx.data(), seg);
        expect(ns == 1 && seg[0][2] == 6 && seg[0][1] == 4 && q.lost == 6 && q.pos[(size_t) q.at(0)] == 6 &&
                   q.next[(size_t) q.at(0)] == 1006,
               "10 rows into 4: rows 6..9");
        r.drop_from(0);
        expect(r.count == 0, "drop all");
        MtpCatchRing z;
        z.init(0);
        expect(z.append(0, 3, nx.data(), seg) == 0 && z.count == 0, "cap 0 keeps nothing");
    }
    // STRATA_GLM_MTP_LEND_MORE: the parse, and the slots over several regions
    {
        expect(!mtp_lend_extra_parse(nullptr, nullptr, nullptr).more, "more: off by default");
        expect(mtp_lend_extra_parse(nullptr, nullptr, "1").more, "more: 1");
        // one region: the same counts as mtp_lend_counts
        const size_t trunk = 11920000;
        std::vector<size_t> st(21, trunk);
        std::vector<int> have(21, 0);
        for (int k = 0; k < 21; ++k) have[(size_t) k] = 204 + (k * 47) % 48;
        const auto gain = [&](int k, int j) { return (1.0 + 0.05 * (21 - k)) / (1.0 + j); };
        const size_t region = 222ull * 14930000ull;
        const std::vector<int> cnt = mtp_lend_counts(region, st, have, 291, gain);
        const auto pl = mtp_lend_place({region}, st, have, 291, gain);
        bool same = pl.size() == cnt.size();
        for (size_t k = 0; same && k < cnt.size(); ++k) same = (int) pl[k].size() == cnt[k];
        expect(same, "one region: mtp_lend_counts' counts");
        // Maya-L's last GPU with MORE: the draft region, the NextN block's weights, the windows' block
        const std::vector<size_t> regs = {region, 173ull << 20, 603ull << 20};
        const auto p3 = mtp_lend_place(regs, st, have, 291, gain);
        std::vector<std::vector<std::pair<size_t, size_t>>> iv(regs.size());
        int tot = 0;
        bool inside = true;
        for (size_t k = 0; k < p3.size(); ++k)
            for (const auto& sl : p3[k]) {
                ++tot;
                inside = inside && sl.region >= 0 && sl.region < (int) regs.size() && sl.off + st[k] <= regs[(size_t) sl.region];
                if (sl.region >= 0 && sl.region < (int) regs.size()) iv[(size_t) sl.region].push_back({sl.off, sl.off + st[k]});
            }
        bool tiled = true;
        for (auto& v : iv) {
            std::sort(v.begin(), v.end());
            for (size_t i = 0; i < v.size(); ++i) tiled = tiled && v[i].first == (i == 0 ? 0 : v[i - 1].second);
        }
        int want = 0;
        for (const size_t r : regs) want += (int) (r / trunk);
        expect(inside && tiled, "several regions: every slot inside one, each region's front tiled, no overlap");
        expect(tot == want, "several regions: as many slots as each region holds");
        // a region smaller than every slot takes none; mixed strides go where they fit
        const auto p4 = mtp_lend_place({50, 250, 100}, {100, 200}, {0, 0}, 291, nullptr);
        int r0 = 0;
        for (const auto& v : p4)
            for (const auto& sl : v) r0 += sl.region == 0;
        expect(r0 == 0, "a region smaller than every slot takes none");
        size_t used1 = 0, used2 = 0;
        for (size_t k = 0; k < p4.size(); ++k)
            for (const auto& sl : p4[k]) (sl.region == 1 ? used1 : used2) += k == 0 ? 100 : 200;
        expect(used1 <= 250 && used2 <= 100 && used1 + used2 >= 300, "mixed strides fill where they fit");
    }
    if (fails == 0) std::printf("mtp_lend_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
