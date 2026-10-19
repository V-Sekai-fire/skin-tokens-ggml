#pragma once

// Rigid assemblies. An anatomical skeleton, a robot or a machine is often
// modelled as many separate closed solids, one per rigid part. Learned skin
// weights spread each part over two to eight joints, so a long bone bends
// "like bubble gum" when the joints turn. A part that must not bend has to
// follow one joint at 100%.
//
// The decision is made for the whole model, not per part: a character with
// skin, hair or clothes has open sheets (hair cards, sleeves, skirts) or one
// large body, and its small closed parts (buttons, cuffs) stay soft, so its
// weights are not touched at all. A model counts as an assembly when almost
// all of its vertices lie in closed parts and no part is a large share of it.
//
// Toon models add an inverted-hull outline around each part: a copy with the
// same vertex and face counts, pushed out along the normals. The outline must
// follow the same joint as the part inside it, or it slides off; each part is
// grouped with the outline that encloses it before the joint is chosen.
//
// Everything works on the welded mesh of the weight transfer (seam twins
// share one vertex), so a test reaches it without the model.

#include <skintokens/skintokens.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <unordered_map>
#include <vector>

namespace skintokens::detail::rigid {

inline constexpr std::uint32_t no_part = std::numeric_limits<std::uint32_t>::max();

struct parts {
    // Part of each welded vertex; `no_part` for a vertex on no face.
    std::vector<std::uint32_t> of;
    std::vector<std::vector<std::uint32_t>> members;
    std::vector<std::size_t> faces;
    // A closed part has every edge on two or more faces (no border).
    std::vector<char> closed;
};

// Connected parts of a welded triangle mesh and whether each one is closed.
inline parts find_parts(std::size_t count, std::span<const std::array<std::uint32_t, 3>> faces) {
    std::vector<std::uint32_t> parent(count);
    std::iota(parent.begin(), parent.end(), 0U);
    const auto root = [&](std::uint32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    std::vector<char> on_face(count, 0);
    std::unordered_map<std::uint64_t, std::uint32_t> edge_uses;
    edge_uses.reserve(faces.size() * 3U);
    for (const auto &face : faces) {
        if (face[0] >= count || face[1] >= count || face[2] >= count) continue;
        for (std::size_t k = 0; k < 3U; ++k) {
            const auto a = face[k], b = face[(k + 1U) % 3U];
            on_face[a] = 1;
            parent[root(a)] = root(b);
            const auto low = std::min(a, b), high = std::max(a, b);
            ++edge_uses[(static_cast<std::uint64_t>(low) << 32U) | high];
        }
    }
    parts out;
    out.of.assign(count, no_part);
    std::vector<std::uint32_t> id_of_root(count, no_part);
    for (std::uint32_t v = 0; v < count; ++v) {
        if (!on_face[v]) continue;
        const auto r = root(v);
        if (id_of_root[r] == no_part) {
            id_of_root[r] = static_cast<std::uint32_t>(out.members.size());
            out.members.emplace_back();
        }
        out.of[v] = id_of_root[r];
        out.members[id_of_root[r]].push_back(v);
    }
    out.faces.assign(out.members.size(), 0U);
    out.closed.assign(out.members.size(), 1);
    for (const auto &face : faces) {
        if (face[0] >= count || face[1] >= count || face[2] >= count) continue;
        ++out.faces[out.of[face[0]]];
    }
    for (const auto &[key, uses] : edge_uses)
        if (uses < 2U) out.closed[out.of[static_cast<std::uint32_t>(key >> 32U)]] = 0;
    return out;
}

struct assembly_rule {
    // Share of vertices that must lie in closed parts.
    double closed_share = 0.9;
    // No single part may hold more than this share of the vertices (a
    // one-piece body is a character, not an assembly).
    double largest_share = 0.25;
    std::size_t min_parts = 8U;
};

inline bool is_assembly(const parts &found, const assembly_rule &rule = {}) {
    if (found.members.size() < rule.min_parts) return false;
    std::size_t total = 0U, closed = 0U, largest = 0U;
    for (std::size_t part = 0; part < found.members.size(); ++part) {
        const auto size = found.members[part].size();
        total += size;
        largest = std::max(largest, size);
        if (found.closed[part]) closed += size;
    }
    if (total == 0U) return false;
    const auto all = static_cast<double>(total);
    return static_cast<double>(closed) >= rule.closed_share * all &&
        static_cast<double>(largest) <= rule.largest_share * all;
}

// Group each part with the outline shell around it: a part with the same
// vertex and face counts whose box encloses this part's box (within 1% of the
// shell's diagonal), whose diagonal is larger, and whose centre lies within
// 10% of its diagonal. Returns a group id per part.
inline std::vector<std::uint32_t> pair_shells(const parts &found, std::span<const vec3> position) {
    const std::size_t count = found.members.size();
    struct box {
        std::array<double, 3> low, high, centre;
        double diagonal;
    };
    std::vector<box> boxes(count);
    for (std::size_t part = 0; part < count; ++part) {
        box b{};
        b.low = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
        b.high = {-b.low[0], -b.low[1], -b.low[2]};
        for (const auto v : found.members[part]) {
            const std::array<double, 3> p{position[v].x, position[v].y, position[v].z};
            for (std::size_t k = 0; k < 3U; ++k) {
                b.low[k] = std::min(b.low[k], p[k]);
                b.high[k] = std::max(b.high[k], p[k]);
                b.centre[k] += p[k];
            }
        }
        const double size = static_cast<double>(std::max<std::size_t>(found.members[part].size(), 1U));
        double d2 = 0.0;
        for (std::size_t k = 0; k < 3U; ++k) {
            b.centre[k] /= size;
            d2 += (b.high[k] - b.low[k]) * (b.high[k] - b.low[k]);
        }
        b.diagonal = std::sqrt(d2);
        boxes[part] = b;
    }
    // Candidates share vertex and face counts, so bucket by them.
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> same_shape;
    for (std::uint32_t part = 0; part < count; ++part)
        same_shape[(static_cast<std::uint64_t>(found.members[part].size()) << 32U) | found.faces[part]].push_back(part);
    std::vector<std::uint32_t> group(count);
    std::iota(group.begin(), group.end(), 0U);
    const auto root = [&](std::uint32_t x) {
        while (group[x] != x) {
            group[x] = group[group[x]];
            x = group[x];
        }
        return x;
    };
    for (const auto &[shape, list] : same_shape) {
        if (list.size() < 2U) continue;
        for (const auto inner : list) {
            const auto &in = boxes[inner];
            std::uint32_t best = no_part;
            double best_distance = std::numeric_limits<double>::infinity();
            for (const auto outer : list) {
                if (outer == inner) continue;
                const auto &out = boxes[outer];
                if (!(out.diagonal > in.diagonal)) continue;
                const double tolerance = 0.01 * out.diagonal;
                bool encloses = true;
                double d2 = 0.0;
                for (std::size_t k = 0; k < 3U; ++k) {
                    encloses = encloses && out.low[k] <= in.low[k] + tolerance && out.high[k] >= in.high[k] - tolerance;
                    d2 += (out.centre[k] - in.centre[k]) * (out.centre[k] - in.centre[k]);
                }
                const double distance = std::sqrt(d2);
                if (!encloses || distance >= 0.1 * out.diagonal || distance >= best_distance) continue;
                best = outer;
                best_distance = distance;
            }
            if (best != no_part) group[root(inner)] = root(best);
        }
    }
    for (std::uint32_t part = 0; part < count; ++part) group[part] = root(part);
    return group;
}

// Give every vertex of each group its group's strongest joint at 100% (the
// joint holding the most weight summed over the group's vertices). `rows` is
// the dense welded weight matrix, `joint_count` floats per vertex. Returns
// the number of parts snapped.
inline std::size_t snap(const parts &found, std::span<const std::uint32_t> group, std::span<float> rows,
    std::size_t joint_count) {
    const std::size_t count = found.members.size();
    if (joint_count == 0U || group.size() != count) return 0U;
    std::vector<double> total(count * joint_count, 0.0);
    for (std::size_t part = 0; part < count; ++part) {
        double *sum = total.data() + static_cast<std::size_t>(group[part]) * joint_count;
        for (const auto v : found.members[part]) {
            const float *row = rows.data() + static_cast<std::size_t>(v) * joint_count;
            for (std::size_t joint = 0; joint < joint_count; ++joint) sum[joint] += row[joint];
        }
    }
    std::size_t snapped = 0U;
    for (std::size_t part = 0; part < count; ++part) {
        const double *sum = total.data() + static_cast<std::size_t>(group[part]) * joint_count;
        const auto strongest = static_cast<std::size_t>(std::max_element(sum, sum + joint_count) - sum);
        if (!(sum[strongest] > 0.0)) continue;
        for (const auto v : found.members[part]) {
            float *row = rows.data() + static_cast<std::size_t>(v) * joint_count;
            std::fill(row, row + joint_count, 0.0F);
            row[strongest] = 1.0F;
        }
        ++snapped;
    }
    return snapped;
}

} // namespace skintokens::detail::rigid
