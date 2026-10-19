// Plays a rigged GLB through two minutes of procedural motion with linear
// blend skinning and reports every vertex that the weights tear away from its
// neighbours.
//
//   skintokens-skin-stress RIGGED.glb [--seconds S] [--fps F] [--degrees A] [--seed N]
//
// Every non-root joint swings about its own random axis by A degrees with its
// own frequency (0.15-0.9 Hz) and phase, so over two minutes each joint passes
// through many poses and many combinations with its neighbours. Checks:
//   invalid  - a vertex whose four weights are not finite, negative, or do not
//              sum to one (1e-3);
//   torn     - a vertex that leaves its neighbours: |p - mean(one ring)| grows
//              past --tear times its rest value in any frame. The rest value
//              is floored at the mean rest edge length there and at 0.5% of
//              the mesh's largest side (--floor), so millimetre fingertip
//              polygons do not turn sub-visible differences into spikes. Correct weights move a vertex
//              with its neighbours, so this stays near one even where linear
//              blend skinning folds a joint; a vertex bound to a remote joint
//              flies off. Edge stretch is printed for reference only;
//   cracked  - coincident vertices (UV / normal seams) that separate by more
//              than 0.1% of the mesh height in any frame.
// Exit 0 only when all three counts are zero.
#include <skintokens/skintokens.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using skintokens::quat;
using skintokens::vec3;

quat axis_angle(vec3 axis, float angle) {
    const float s = std::sin(angle * 0.5F);
    return {axis.x * s, axis.y * s, axis.z * s, std::cos(angle * 0.5F)};
}

float length(vec3 a, vec3 b) {
    const float x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
    return std::sqrt(x * x + y * y + z * z);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: skintokens-skin-stress RIGGED.glb [--seconds S] [--fps F]"
            " [--degrees A] [--seed N] [--tear R] [--floor F]\n");
        return 2;
    }
    float seconds = 120.0F, fps = 30.0F, degrees = 60.0F, tear = 3.0F, floor_fraction = 0.005F;
    unsigned seed = 1U;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string_view key{argv[i]};
        const float value = std::strtof(argv[i + 1], nullptr);
        if (key == "--seconds") seconds = value;
        else if (key == "--fps") fps = value;
        else if (key == "--degrees") degrees = value;
        else if (key == "--seed") seed = static_cast<unsigned>(value);
        else if (key == "--tear") tear = value;
        else if (key == "--floor") floor_fraction = value;
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    auto asset = skintokens::load_skinned_glb_file(argv[1]);
    if (!asset) {
        std::fprintf(stderr, "FAIL load: %s\n", asset.error().message.c_str());
        return 1;
    }
    const auto &geometry = asset->geometry;
    const auto &binding = asset->binding;
    const std::size_t n = geometry.vertices.size();
    const std::size_t joints = binding.rig.names.size();

    // 1. The weights themselves.
    std::size_t invalid = 0U;
    for (std::size_t v = 0; v < n; ++v) {
        float sum = 0.0F;
        bool bad = false;
        for (std::size_t s = 0; s < 4U; ++s) {
            const float w = binding.weights[v][s];
            bad = bad || !std::isfinite(w) || w < 0.0F || binding.joints[v][s] >= joints;
            sum += w;
        }
        if (bad || std::abs(sum - 1.0F) > 1.0e-3F) ++invalid;
    }

    // 2. Welded edges and coincident vertex groups.
    vec3 low = geometry.vertices.front(), high = low;
    for (const auto p : geometry.vertices) {
        low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
        high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
    }
    const float extent = std::max({high.x - low.x, high.y - low.y, high.z - low.z});
    const float weld = extent * 1.0e-6F;
    std::map<std::tuple<long long, long long, long long>, std::uint32_t> cells;
    std::vector<std::uint32_t> welded(n);
    std::vector<std::vector<std::uint32_t>> twins;
    for (std::size_t v = 0; v < n; ++v) {
        const auto p = geometry.vertices[v];
        const auto key = std::make_tuple(std::llround(p.x / weld), std::llround(p.y / weld), std::llround(p.z / weld));
        const auto [it, fresh] = cells.emplace(key, static_cast<std::uint32_t>(twins.size()));
        if (fresh) twins.emplace_back();
        welded[v] = it->second;
        twins[it->second].push_back(static_cast<std::uint32_t>(v));
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
    for (const auto &f : geometry.faces)
        for (std::size_t k = 0; k < 3U; ++k) {
            auto a = f[k], b = f[(k + 1U) % 3U];
            if (welded[a] == welded[b]) continue;
            if (a > b) std::swap(a, b);
            edges.emplace_back(a, b);
        }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    std::vector<float> rest(edges.size());
    for (std::size_t e = 0; e < edges.size(); ++e)
        rest[e] = std::max(length(geometry.vertices[edges[e].first], geometry.vertices[edges[e].second]), weld);
    // One ring per welded vertex (through one representative), for the spike test.
    const std::size_t wcount = twins.size();
    std::vector<std::vector<std::uint32_t>> ring(wcount);
    for (const auto &[a, b] : edges) {
        ring[welded[a]].push_back(twins[welded[b]][0]);
        ring[welded[b]].push_back(twins[welded[a]][0]);
    }
    for (auto &list : ring) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    const auto offset = [&](const std::vector<vec3> &p, std::size_t w) {
        vec3 mean{0.0F, 0.0F, 0.0F};
        for (const auto other : ring[w]) {
            mean.x += p[other].x;
            mean.y += p[other].y;
            mean.z += p[other].z;
        }
        const float k = static_cast<float>(ring[w].size());
        mean = {mean.x / k, mean.y / k, mean.z / k};
        return length(p[twins[w][0]], mean);
    };
    std::vector<float> rest_offset(wcount, 0.0F);
    {
        std::vector<float> edge_sum(wcount, 0.0F);
        std::vector<float> edge_count(wcount, 0.0F);
        for (std::size_t e = 0; e < edges.size(); ++e) {
            for (const auto v : {edges[e].first, edges[e].second}) {
                edge_sum[welded[v]] += rest[e];
                edge_count[welded[v]] += 1.0F;
            }
        }
        for (std::size_t w = 0; w < wcount; ++w)
            if (!ring[w].empty())
                rest_offset[w] = std::max({offset(geometry.vertices, w), edge_sum[w] / edge_count[w], floor_fraction * extent});
    }

    // 3. The motion.
    skintokens::motion clip;
    clip.rig = binding.rig;
    clip.frames_per_second = fps;
    clip.frames = static_cast<std::size_t>(seconds * fps);
    clip.root_translations.assign(clip.frames, binding.rig.rest_positions.front());
    clip.local_rotations.assign(clip.frames * joints, quat{});
    std::mt19937 random{seed};
    std::normal_distribution<float> normal{0.0F, 1.0F};
    std::uniform_real_distribution<float> uniform{0.0F, 1.0F};
    const float amplitude = degrees * 3.14159265F / 180.0F;
    for (std::size_t j = 1; j < joints; ++j) {
        vec3 axis{normal(random), normal(random), normal(random)};
        const float l = std::max(std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z), 1.0e-6F);
        axis = {axis.x / l, axis.y / l, axis.z / l};
        const float hz = 0.15F + 0.75F * uniform(random);
        const float phase = 6.2831853F * uniform(random);
        for (std::size_t f = 0; f < clip.frames; ++f) {
            const float t = static_cast<float>(f) / fps;
            clip.local_rotations[f * joints + j] = axis_angle(axis, amplitude * std::sin(6.2831853F * hz * t + phase));
        }
    }

    // 4. Play it.
    std::vector<float> worst(n, 1.0F), spike(wcount, 0.0F);
    std::vector<char> torn(n, 0), cracked(n, 0);
    float worst_crack = 0.0F;
    std::vector<float> frame_max;
    frame_max.reserve(clip.frames);
    for (std::size_t f = 0; f < clip.frames; ++f) {
        auto posed = skintokens::deform_vertices(geometry, binding, clip, f);
        if (!posed) {
            std::fprintf(stderr, "FAIL frame %zu: %s\n", f, posed.error().message.c_str());
            return 1;
        }
        float peak = 1.0F, spiked = 0.0F;
        for (std::size_t e = 0; e < edges.size(); ++e) {
            const auto [a, b] = edges[e];
            const float ratio = length((*posed)[a], (*posed)[b]) / rest[e];
            peak = std::max(peak, ratio);
            worst[a] = std::max(worst[a], ratio);
            worst[b] = std::max(worst[b], ratio);
        }
        for (std::size_t w = 0; w < wcount; ++w) {
            if (ring[w].empty()) continue;
            const float ratio = offset(*posed, w) / rest_offset[w];
            spike[w] = std::max(spike[w], ratio);
            if (ratio > tear) {
                for (const auto v : twins[w]) torn[v] = 1;
                spiked = std::max(spiked, ratio);
            }
        }
        (void) peak;
        frame_max.push_back(spiked);
        for (const auto &group : twins) {
            for (std::size_t k = 1; k < group.size(); ++k) {
                const float gap = length((*posed)[group[0]], (*posed)[group[k]]) / extent;
                worst_crack = std::max(worst_crack, gap);
                if (gap > 1.0e-3F) cracked[group[0]] = cracked[group[k]] = 1;
            }
        }
    }
    std::vector<float> sorted = worst;
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double q) { return sorted[static_cast<std::size_t>(q * static_cast<double>(n - 1U))]; };
    std::size_t torn_count = 0U, cracked_count = 0U, frames_torn = 0U;
    for (std::size_t v = 0; v < n; ++v) {
        torn_count += torn[v];
        cracked_count += cracked[v];
    }
    for (const float value : frame_max) frames_torn += value > tear ? 1U : 0U;
    std::printf("STRESS file=%s vertices=%zu welded=%zu joints=%zu frames=%zu seconds=%.0f degrees=%.0f floor=%.3f\n",
        argv[1], n, twins.size(), joints, clip.frames, seconds, degrees, floor_fraction);
    std::printf("STRESS invalid=%zu torn=%zu (tear>%.1fx) frames_torn=%zu cracked=%zu worst_crack=%.2e\n",
        invalid, torn_count, tear, frames_torn, cracked_count, worst_crack);
    std::printf("STRESS edge-stretch p50=%.3f p99=%.3f p999=%.3f max=%.3f\n", pct(0.5), pct(0.99), pct(0.999), sorted.back());
    std::vector<float> spikes = spike;
    std::sort(spikes.begin(), spikes.end());
    const auto spct = [&](double q) { return spikes[static_cast<std::size_t>(q * static_cast<double>(wcount - 1U))]; };
    std::printf("STRESS spike p50=%.3f p99=%.3f p999=%.3f max=%.3f\n", spct(0.5), spct(0.99), spct(0.999), spikes.back());
    if (const char *list = std::getenv("STRESS_LIST")) {
        std::vector<std::uint32_t> order(wcount);
        std::iota(order.begin(), order.end(), 0U);
        std::sort(order.begin(), order.end(), [&](auto a, auto b) { return spike[a] > spike[b]; });
        for (std::size_t k = 0; k < std::min<std::size_t>(std::strtoul(list, nullptr, 10), wcount); ++k) {
            const auto v = twins[order[k]][0];
            const auto p = geometry.vertices[v];
            std::printf("WORST v=%u spike=%.2f at=(%.3f %.3f %.3f) w=", v, spike[order[k]], p.x, p.y, p.z);
            for (std::size_t s4 = 0; s4 < 4U; ++s4)
                std::printf("%s:%.2f ", binding.rig.names[binding.joints[v][s4]].c_str(), binding.weights[v][s4]);
            std::printf("\n");
        }
    }
    const bool pass = invalid == 0U && torn_count == 0U && cracked_count == 0U;
    std::printf("%s\n", pass ? "STRESS-PASS" : "STRESS-FAIL");
    return pass ? 0 : 3;
}
