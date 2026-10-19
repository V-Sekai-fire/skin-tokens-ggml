// cut_clip on a made-up clip: a 3-joint chain that swings with a period of 33 frames, ramps up
// from rest over the first second and back to rest over the last, and walks forward at 1.5 m/s
// while it swings. The loop must find the period, end where it starts (frame 0 repeated), and
// stay on the spot; the one-shot must drop the still start. Controls: a still clip keeps the
// requested length, and a still clip shorter than that length fails.
#include <skintokens/skintokens.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

namespace {
int failures = 0;
void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

skintokens::motion swing(std::size_t frames, float period, bool still, float lead = 0.0F) {
    skintokens::motion m;
    m.frames = frames;
    m.frames_per_second = 30.0F;
    m.rig.names = {"a", "b", "c"};
    m.rig.parents = {-1, 0, 1};
    m.rig.rest_positions = {{0, 1, 0}, {0, 1.5F, 0}, {0, 2, 0}};
    for (std::size_t f = 0; f < frames; ++f) {
        const float t = static_cast<float>(f) - lead;
        float amp = 1.0F;
        if (t < 30.0F) amp = std::max(0.0F, t / 30.0F);
        if (t > static_cast<float>(frames) - 45.0F) amp = std::max(0.0F, (static_cast<float>(frames) - 15.0F - t) / 30.0F);
        if (still) amp = 0.0F;
        const float speed = still ? 0.0F : 1.5F * amp;
        static float z = 0.0F;
        if (f == 0) z = 0.0F;
        z += speed / 30.0F;
        m.root_translations.push_back({0, 1, z});
        for (int j = 0; j < 3; ++j) {
            const float a = 0.4F * amp * std::sin(2.0F * std::numbers::pi_v<float> * t / period + static_cast<float>(j));
            m.local_rotations.push_back({std::sin(a / 2), 0, 0, std::cos(a / 2)});
        }
    }
    return m;
}
} // namespace

int main() {
    using skintokens::clip_cut;
    auto loop = skintokens::cut_clip(swing(180, 33.0F, false), clip_cut::loop, 1.1F);
    check(loop.has_value(), "loop cut succeeds");
    if (loop) {
        const auto & [m, r] = *loop;
        std::printf("  start %zu frames %zu seam %.3f deg travel %.3f m/s\n", r.start, r.frames,
                    r.seam_after_max_degrees, r.travel_removed_metres_per_second);
        check(r.frames == 33U, "loop length is the swing period (33 frames)");
        check(r.start >= 30U && r.start + r.frames <= 135U, "loop lies in the steady part");
        check(m.frames == r.frames + 1U, "frame 0 is repeated at the end");
        const auto & q0 = m.local_rotations.front(), & qe = m.local_rotations[r.frames * 3U];
        check(q0.x == qe.x && q0.w == qe.w, "the last frame equals the first");
        check(std::abs(m.root_translations.back().z - m.root_translations.front().z) < 1e-4F, "the root ends where it starts");
        check(std::abs(r.travel_removed_metres_per_second - 1.5F) < 0.05F, "removed travel is the walk speed");
        check(r.seam_after_max_degrees <= 1.05F * r.largest_step_degrees, "the wrap is no bigger than one normal frame step");
    }
    // 40 still frames first: the one-shot starts a few frames before the motion does.
    auto once = skintokens::cut_clip(swing(180, 33.0F, false, 40.0F), clip_cut::once, 2.0F);
    if (once) std::printf("  once start %zu frames %zu\n", once->second.start, once->second.frames);
    check(once && once->second.start >= 34U && once->second.start <= 46U, "one-shot starts just before the motion");
    auto still = skintokens::cut_clip(swing(180, 33.0F, true), clip_cut::loop, 4.0F);
    check(still && still->second.frames == 120U, "control: a still clip keeps the requested 4 s");
    auto none = skintokens::cut_clip(swing(60, 33.0F, true), clip_cut::loop, 4.0F);
    check(!none.has_value(), "control: a 2 s still clip cannot give a 4 s loop");
    std::printf("%s\n", failures ? "MOTION-CLIP FAIL" : "MOTION-CLIP OK");
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
