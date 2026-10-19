// Twist bones for a generated humanoid rig, after SOMA-X's procedural twist joints: four helpers
// on each upper arm, forearm, thigh and shin at 0.05, 1/3, 2/3 and 0.95 of the segment. The
// segment's skin weight is split among them by where each vertex lies along the segment. That
// split is one ggml graph, so it runs on the rig's own device (ggml-rd inside a sandbox guest).
#include "internal.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace skintokens {
namespace {

// SOMA30 indices: {inherited-from, start, end}. Upper segments are "reverse" (SOMA-X): their
// helpers near the start undo the start joint's own twist, so the shoulder and hip do not wring.
struct segment_role { int parent; int start; int end; bool reverse; };
constexpr std::array<segment_role, 8> segment_roles{{
    {10, 11, 12, true}, {11, 12, 13, false},   // left upper arm, forearm
    {16, 17, 18, true}, {17, 18, 19, false},   // right upper arm, forearm
    {0, 22, 23, true}, {22, 23, 24, false},    // left thigh, shin
    {0, 26, 27, true}, {26, 27, 28, false},    // right thigh, shin
}};

vec3 sub3(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
float dot3(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

} // namespace

const std::array<float, twist_helpers_per_segment> & twist_fractions() {
    static const std::array<float, twist_helpers_per_segment> values{0.05F, 1.0F / 3.0F, 2.0F / 3.0F, 0.95F};
    return values;
}

result<twist_rig> add_twist_joints(const mesh & geometry, const skin & binding, const runtime_options & runtime) try {
    const std::size_t vertex_count = geometry.vertices.size();
    if (binding.joints.size() != vertex_count || binding.weights.size() != vertex_count)
        return std::unexpected(detail::fail(error_code::invalid_argument, "twist joints: binding does not match the mesh"));
    auto mapping = match_generated_humanoid(binding.rig);
    if (!mapping) return std::unexpected(mapping.error());

    twist_rig out;
    out.binding = binding;
    skeleton & rig = out.binding.rig;
    const auto & fractions = twist_fractions();
    for (const auto & role : segment_roles) {
        const std::int32_t parent = mapping->soma_to_generated[static_cast<std::size_t>(role.parent)];
        const std::int32_t start = mapping->soma_to_generated[static_cast<std::size_t>(role.start)];
        const std::int32_t end = mapping->soma_to_generated[static_cast<std::size_t>(role.end)];
        if (parent < 0 || start < 0 || end < 0) continue;
        const vec3 a = rig.rest_positions[static_cast<std::size_t>(start)];
        const vec3 d = sub3(rig.rest_positions[static_cast<std::size_t>(end)], a);
        if (dot3(d, d) < 1e-12F) continue;
        twist_segment segment{parent, start, end, role.reverse, {}};
        for (std::size_t k = 0; k < twist_helpers_per_segment; ++k) {
            segment.helpers[k] = static_cast<std::int32_t>(rig.names.size());
            rig.names.push_back(rig.names[static_cast<std::size_t>(start)] + "Twist" + std::to_string(k + 1U));
            rig.parents.push_back(start);
            rig.rest_positions.push_back({a.x + fractions[k] * d.x, a.y + fractions[k] * d.y, a.z + fractions[k] * d.z});
        }
        out.segments.push_back(segment);
    }
    if (out.segments.empty()) return out;
    if (rig.names.size() > 65535U)
        return std::unexpected(detail::fail(error_code::limit_exceeded, "twist joints: too many joints for JOINTS_0"));

    // Tent shares along each segment. Each side of a tent is an affine function of the vertex
    // (x, y, z, 1): rise_k = (t - f[k-1]) / (f[k] - f[k-1]), fall_k = (f[k+1] - t) / (f[k+1] - f[k]),
    // with t = (v - a).d / |d|^2. The first helper has no rise and the last no fall (they hold the
    // ends), so share = clamp(min(rise, fall), 0, 1) over two matrices. The shares sum to 1.
    const std::size_t rows = out.segments.size() * twist_helpers_per_segment;
    std::vector<float> rise(rows * 4U), fall(rows * 4U);
    for (std::size_t s = 0; s < out.segments.size(); ++s) {
        const vec3 a = rig.rest_positions[static_cast<std::size_t>(out.segments[s].start)];
        const vec3 d = sub3(rig.rest_positions[static_cast<std::size_t>(out.segments[s].end)], a);
        const float inv = 1.0F / dot3(d, d);
        const std::array<float, 4> t_of{d.x * inv, d.y * inv, d.z * inv, -dot3(a, d) * inv};  // t = t_of . (x,y,z,1)
        for (std::size_t k = 0; k < twist_helpers_per_segment; ++k) {
            const std::size_t row = s * twist_helpers_per_segment + k;
            const auto affine = [&](float slope, float offset, std::vector<float> & m) {
                for (std::size_t c = 0; c < 4U; ++c) m[row * 4U + c] = slope * t_of[c] + (c == 3U ? offset : 0.0F);
            };
            const bool first = k == 0U, last = k + 1U == twist_helpers_per_segment;
            const float lo = first ? 0.0F : fractions[k - 1U], hi = last ? 1.0F : fractions[k + 1U];
            // fall: (hi - t) / (hi - f[k]); rise: (t - lo) / (f[k] - lo). An end side is the other side.
            if (!last) affine(-1.0F / (hi - fractions[k]), hi / (hi - fractions[k]), fall);
            if (!first) affine(1.0F / (fractions[k] - lo), -lo / (fractions[k] - lo), rise);
            if (first) std::copy_n(fall.begin() + static_cast<std::ptrdiff_t>(row * 4U), 4, rise.begin() + static_cast<std::ptrdiff_t>(row * 4U));
            if (last) std::copy_n(rise.begin() + static_cast<std::ptrdiff_t>(row * 4U), 4, fall.begin() + static_cast<std::ptrdiff_t>(row * 4U));
        }
    }
    std::vector<float> points(vertex_count * 4U);
    for (std::size_t v = 0; v < vertex_count; ++v) {
        points[v * 4U] = geometry.vertices[v].x; points[v * 4U + 1U] = geometry.vertices[v].y;
        points[v * 4U + 2U] = geometry.vertices[v].z; points[v * 4U + 3U] = 1.0F;
    }

    auto backend = detail::make_backend(runtime);
    if (!backend) return std::unexpected(backend.error());
    auto * context = ggml_init({16ULL << 20U, nullptr, true});
    if (context == nullptr) return std::unexpected(detail::fail(error_code::allocation, "cannot create twist graph"));
    auto cleanup = std::unique_ptr<ggml_context, decltype(&ggml_free)>(context, ggml_free);
    const auto R = static_cast<std::int64_t>(rows), N = static_cast<std::int64_t>(vertex_count);
    auto * rise_m = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, R);
    auto * fall_m = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, R);
    auto * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, N);
    ggml_set_input(rise_m); ggml_set_input(fall_m); ggml_set_input(input);
    auto * p = ggml_mul_mat(context, rise_m, input);  // ne0 = rows, ne1 = vertices
    auto * q = ggml_mul_mat(context, fall_m, input);
    auto * lower = ggml_scale(context, ggml_sub(context, ggml_add(context, p, q), ggml_abs(context, ggml_sub(context, p, q))), 0.5F);
    auto * share = ggml_clamp(context, lower, 0.0F, 1.0F);
    ggml_set_output(share);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, share);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type((*backend)->value));
    if (allocator == nullptr || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator != nullptr) ggml_gallocr_free(allocator);
        return std::unexpected(detail::fail(error_code::allocation, "cannot allocate twist graph"));
    }
    auto allocator_cleanup = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(allocator, ggml_gallocr_free);
    ggml_backend_tensor_set(rise_m, rise.data(), 0, rise.size() * sizeof(float));
    ggml_backend_tensor_set(fall_m, fall.data(), 0, fall.size() * sizeof(float));
    ggml_backend_tensor_set(input, points.data(), 0, points.size() * sizeof(float));
    if (ggml_backend_graph_compute((*backend)->value, graph) != GGML_STATUS_SUCCESS)
        return std::unexpected(detail::fail(error_code::compute, "twist graph execution failed"));
    out.shares.resize(rows * vertex_count);
    ggml_backend_tensor_get(share, out.shares.data(), 0, out.shares.size() * sizeof(float));

    // A start joint's weight moves to its helpers by share; each vertex keeps its four largest.
    std::map<std::int32_t, std::size_t> segment_of_start;
    for (std::size_t s = 0; s < out.segments.size(); ++s) segment_of_start[out.segments[s].start] = s;
    for (std::size_t v = 0; v < vertex_count; ++v) {
        std::map<std::int32_t, float> total;
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const float w = binding.weights[v][slot];
            if (w <= 0.0F) continue;
            const auto joint = static_cast<std::int32_t>(binding.joints[v][slot]);
            const auto found = segment_of_start.find(joint);
            if (found == segment_of_start.end()) { total[joint] += w; continue; }
            const std::size_t s = found->second;
            for (std::size_t k = 0; k < twist_helpers_per_segment; ++k) {
                const float part = w * out.shares[v * rows + s * twist_helpers_per_segment + k];
                if (part > 0.0F) total[out.segments[s].helpers[k]] += part;
            }
        }
        std::vector<std::pair<float, std::int32_t>> ranked;
        for (const auto & [joint, w] : total) ranked.push_back({w, joint});
        std::sort(ranked.begin(), ranked.end(), [](const auto & l, const auto & r) {
            return l.first != r.first ? l.first > r.first : l.second < r.second;
        });
        float kept = 0.0F;
        for (std::size_t slot = 0; slot < 4U && slot < ranked.size(); ++slot) kept += ranked[slot].first;
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const bool has = slot < ranked.size() && kept > 0.0F;
            out.binding.joints[v][slot] = has ? static_cast<std::uint16_t>(ranked[slot].second) : 0U;
            out.binding.weights[v][slot] = has ? ranked[slot].first / kept : 0.0F;
        }
    }
    return out;
} catch (const std::exception & exception) {
    return std::unexpected(detail::fail(error_code::compute, exception.what()));
}

namespace detail {

bool is_twist_helper(const skeleton & rig, std::size_t joint) {
    const std::int32_t parent = joint < rig.parents.size() ? rig.parents[joint] : -1;
    if (parent < 0) return false;
    const std::string & name = rig.names[joint];
    const std::string stem = rig.names[static_cast<std::size_t>(parent)] + "Twist";
    if (name.size() != stem.size() + 1U || name.compare(0, stem.size(), stem) != 0) return false;
    if (name.back() < '1' || name.back() > static_cast<char>('0' + twist_helpers_per_segment)) return false;
    for (std::size_t child = 0; child < rig.parents.size(); ++child)
        if (rig.parents[child] == static_cast<std::int32_t>(joint)) return false;
    return true;
}

skeleton without_twist_helpers(const skeleton & rig) {
    skeleton out;
    std::vector<std::int32_t> index_of(rig.names.size(), -1);
    for (std::size_t joint = 0; joint < rig.names.size(); ++joint) {
        if (is_twist_helper(rig, joint)) continue;
        index_of[joint] = static_cast<std::int32_t>(out.names.size());
        out.names.push_back(rig.names[joint]);
        const std::int32_t parent = rig.parents[joint];
        out.parents.push_back(parent < 0 ? -1 : index_of[static_cast<std::size_t>(parent)]);
        out.rest_positions.push_back(rig.rest_positions[joint]);
    }
    return out;
}

} // namespace detail
} // namespace skintokens
