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

namespace {

constexpr std::size_t role_count = segment_roles.size();

result<twist_rig> build_twist(const mesh & geometry, const skin & binding, const runtime_options & runtime,
                              const std::array<bool, role_count> & enabled, std::vector<std::size_t> & role_of) try {
    const std::size_t vertex_count = geometry.vertices.size();
    if (binding.joints.size() != vertex_count || binding.weights.size() != vertex_count)
        return std::unexpected(detail::fail(error_code::invalid_argument, "twist joints: binding does not match the mesh"));
    auto mapping = match_generated_humanoid(binding.rig);
    if (!mapping) return std::unexpected(mapping.error());

    twist_rig out;
    out.binding = binding;
    skeleton & rig = out.binding.rig;
    const auto & fractions = twist_fractions();
    role_of.clear();
    for (std::size_t r = 0; r < role_count; ++r) {
        if (!enabled[r]) continue;
        const auto & role = segment_roles[r];
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
        role_of.push_back(r);
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

    // A start joint's weight moves to its helpers by share. The tents overlap in pairs, so a start
    // joint becomes at most two helpers. No other joint's weight is ever dropped: when a vertex has
    // no free slot for a second helper, the smaller helper share merges into the larger one. A VRM
    // viewer skins with four weights per vertex, so this keeps every vertex inside four slots.
    std::map<std::int32_t, std::size_t> segment_of_start;
    for (std::size_t s = 0; s < out.segments.size(); ++s) segment_of_start[out.segments[s].start] = s;
    for (std::size_t v = 0; v < vertex_count; ++v) {
        std::vector<std::pair<std::int32_t, float>> kept;                     // joints that stay
        std::vector<std::vector<std::pair<std::int32_t, float>>> splits;     // per start joint: helpers
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const float w = binding.weights[v][slot];
            if (w <= 0.0F) continue;
            const auto joint = static_cast<std::int32_t>(binding.joints[v][slot]);
            const auto found = segment_of_start.find(joint);
            if (found == segment_of_start.end()) { kept.push_back({joint, w}); continue; }
            const std::size_t s = found->second;
            std::vector<std::pair<std::int32_t, float>> parts;
            for (std::size_t k = 0; k < twist_helpers_per_segment; ++k) {
                const float part = w * out.shares[v * rows + s * twist_helpers_per_segment + k];
                if (part > 0.0F) parts.push_back({out.segments[s].helpers[k], part});
            }
            if (parts.empty()) { kept.push_back({joint, w}); continue; }
            splits.push_back(std::move(parts));
        }
        std::size_t used = kept.size();
        for (const auto & parts : splits) used += parts.size();
        for (auto & parts : splits) {
            if (used <= 4U) break;
            if (parts.size() < 2U) continue;
            std::sort(parts.begin(), parts.end(), [](const auto & l, const auto & r) {
                return l.second != r.second ? l.second > r.second : l.first < r.first;
            });
            float rest = 0.0F;
            for (std::size_t k = 1; k < parts.size(); ++k) rest += parts[k].second;
            used -= parts.size() - 1U;
            parts.resize(1U);
            parts[0].second += rest;
        }
        for (const auto & parts : splits) kept.insert(kept.end(), parts.begin(), parts.end());
        std::sort(kept.begin(), kept.end(), [](const auto & l, const auto & r) {
            return l.second != r.second ? l.second > r.second : l.first < r.first;
        });
        float total = 0.0F;
        for (std::size_t slot = 0; slot < 4U && slot < kept.size(); ++slot) total += kept[slot].second;
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const bool has = slot < kept.size() && total > 0.0F;
            out.binding.joints[v][slot] = has ? static_cast<std::uint16_t>(kept[slot].first) : 0U;
            out.binding.weights[v][slot] = has ? kept[slot].second / total : 0.0F;
        }
    }
    return out;
} catch (const std::exception & exception) {
    return std::unexpected(detail::fail(error_code::compute, exception.what()));
}

// The 3x4 rigid transform turning by angle about the line through c along unit axis u.
std::array<float, 12> turn_about(vec3 c, vec3 u, float angle) {
    const float co = std::cos(angle), si = std::sin(angle), t = 1.0F - co;
    const float R[3][3] = {{t * u.x * u.x + co, t * u.x * u.y - si * u.z, t * u.x * u.z + si * u.y},
                           {t * u.x * u.y + si * u.z, t * u.y * u.y + co, t * u.y * u.z - si * u.x},
                           {t * u.x * u.z - si * u.y, t * u.y * u.z + si * u.x, t * u.z * u.z + co}};
    const float cv[3] = {c.x, c.y, c.z};
    std::array<float, 12> m{};
    for (int r = 0; r < 3; ++r) {
        float rc = 0.0F;
        for (int k = 0; k < 3; ++k) { m[static_cast<std::size_t>(r * 4 + k)] = R[r][k]; rc += R[r][k] * cv[k]; }
        m[static_cast<std::size_t>(r * 4 + 3)] = cv[r] - rc;
    }
    return m;
}

bool under(const skeleton & rig, std::int32_t joint, std::int32_t root) {
    for (std::size_t guard = 0; joint >= 0 && guard <= rig.parents.size(); ++guard) {
        if (joint == root) return true;
        joint = rig.parents[static_cast<std::size_t>(joint)];
    }
    return false;
}

constexpr std::array<float, 4> sweep_angles{-1.5707963F, -0.7853982F, 0.7853982F, 1.5707963F};

// Skin every vertex under every pose on the device: blend = deltas . W (a matrix product over
// joints), then position = blend . (x, y, z, 1). poses holds 12 floats per joint per pose.
result<std::vector<float>> skin_poses(const mesh & geometry, const skin & binding, const std::vector<float> & poses,
                                      std::size_t pose_count, const runtime_options & runtime) {
    const std::size_t J = binding.rig.names.size(), N = geometry.vertices.size();
    std::vector<float> w(J * N, 0.0F), points(N * 4U);
    for (std::size_t v = 0; v < N; ++v) {
        for (std::size_t slot = 0; slot < 4U; ++slot)
            if (binding.weights[v][slot] > 0.0F) w[v * J + binding.joints[v][slot]] += binding.weights[v][slot];
        points[v * 4U] = geometry.vertices[v].x; points[v * 4U + 1U] = geometry.vertices[v].y;
        points[v * 4U + 2U] = geometry.vertices[v].z; points[v * 4U + 3U] = 1.0F;
    }
    std::vector<float> d(J * 12U * pose_count);  // ne0 = joint, ne1 = (pose, entry)
    for (std::size_t q = 0; q < pose_count; ++q)
        for (std::size_t j = 0; j < J; ++j)
            for (std::size_t c = 0; c < 12U; ++c) d[(q * 12U + c) * J + j] = poses[(q * J + j) * 12U + c];
    auto backend = detail::make_backend(runtime);
    if (!backend) return std::unexpected(backend.error());
    auto * context = ggml_init({16ULL << 20U, nullptr, true});
    if (context == nullptr) return std::unexpected(detail::fail(error_code::allocation, "cannot create twist check graph"));
    auto cleanup = std::unique_ptr<ggml_context, decltype(&ggml_free)>(context, ggml_free);
    const auto Ji = static_cast<std::int64_t>(J), Ni = static_cast<std::int64_t>(N), Qi = static_cast<std::int64_t>(pose_count);
    auto * deltas = ggml_new_tensor_2d(context, GGML_TYPE_F32, Ji, 12 * Qi);
    auto * weights = ggml_new_tensor_2d(context, GGML_TYPE_F32, Ji, Ni);
    auto * input = ggml_new_tensor_4d(context, GGML_TYPE_F32, 4, 1, 1, Ni);
    ggml_set_input(deltas); ggml_set_input(weights); ggml_set_input(input);
    auto * blend = ggml_mul_mat(context, deltas, weights);                 // (12Q, N)
    auto * rows = ggml_reshape_4d(context, blend, 4, 3, Qi, Ni);
    auto * skinned = ggml_sum_rows(context, ggml_mul(context, rows, input));  // (1, 3, Q, N)
    ggml_set_output(skinned);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, skinned);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type((*backend)->value));
    if (allocator == nullptr || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator != nullptr) ggml_gallocr_free(allocator);
        return std::unexpected(detail::fail(error_code::allocation, "cannot allocate twist check graph"));
    }
    auto allocator_cleanup = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(allocator, ggml_gallocr_free);
    ggml_backend_tensor_set(deltas, d.data(), 0, d.size() * sizeof(float));
    ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(input, points.data(), 0, points.size() * sizeof(float));
    if (ggml_backend_graph_compute((*backend)->value, graph) != GGML_STATUS_SUCCESS)
        return std::unexpected(detail::fail(error_code::compute, "twist check graph execution failed"));
    std::vector<float> out(3U * pose_count * N);
    ggml_backend_tensor_get(skinned, out.data(), 0, out.size() * sizeof(float));
    return out;  // index (v * Q + q) * 3 + axis
}

// Per segment: the worst 1st-percentile triangle area ratio over the sweep, for the plain binding
// and the twisted one. The twisted rig turns its helpers by the clip rule; the plain one cannot.
result<std::vector<twist_check>> check_segments(const mesh & geometry, const skin & plain, const twist_rig & twisted,
                                                const runtime_options & runtime) {
    const std::size_t S = twisted.segments.size(), A = sweep_angles.size(), Q = S * A;
    const auto & fractions = twist_fractions();
    const skeleton & tr = twisted.binding.rig;
    const std::size_t Jp = plain.rig.names.size(), Jt = tr.names.size();
    std::vector<float> plain_poses(Q * Jp * 12U), twist_poses(Q * Jt * 12U);
    const std::array<float, 12> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    for (std::size_t s = 0; s < S; ++s) {
        const auto & seg = twisted.segments[s];
        const vec3 a = tr.rest_positions[static_cast<std::size_t>(seg.start)];
        vec3 u = sub3(tr.rest_positions[static_cast<std::size_t>(seg.end)], a);
        const float length = std::sqrt(dot3(u, u));
        u = {u.x / length, u.y / length, u.z / length};
        const std::int32_t driver = seg.reverse ? seg.start : seg.end;
        for (std::size_t i = 0; i < A; ++i) {
            const std::size_t q = s * A + i;
            const auto full = turn_about(a, u, sweep_angles[i]);
            for (std::size_t j = 0; j < Jp; ++j) {
                const auto & m = under(plain.rig, static_cast<std::int32_t>(j), driver) ? full : identity;
                std::copy(m.begin(), m.end(), plain_poses.begin() + static_cast<std::ptrdiff_t>((q * Jp + j) * 12U));
            }
            for (std::size_t j = 0; j < Jt; ++j) {
                std::array<float, 12> m = under(tr, static_cast<std::int32_t>(j), driver) ? full : identity;
                for (std::size_t k = 0; k < twist_helpers_per_segment; ++k)
                    if (seg.helpers[k] == static_cast<std::int32_t>(j)) m = turn_about(a, u, fractions[k] * sweep_angles[i]);
                std::copy(m.begin(), m.end(), twist_poses.begin() + static_cast<std::ptrdiff_t>((q * Jt + j) * 12U));
            }
        }
    }
    auto p = skin_poses(geometry, plain, plain_poses, Q, runtime);
    if (!p) return std::unexpected(p.error());
    auto t = skin_poses(geometry, twisted.binding, twist_poses, Q, runtime);
    if (!t) return std::unexpected(t.error());
    const auto area = [](const float * x, const float * y, const float * z) {
        const float e1[3] = {y[0] - x[0], y[1] - x[1], y[2] - x[2]}, e2[3] = {z[0] - x[0], z[1] - x[1], z[2] - x[2]};
        const float c[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        return 0.5F * std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    };
    std::vector<twist_check> checks;
    for (std::size_t s = 0; s < S; ++s) {
        const auto & seg = twisted.segments[s];
        const std::int32_t driver = seg.reverse ? seg.start : seg.end;
        // Triangles the sweep moves: any corner weighted on the start joint or the driver's subtree.
        std::vector<std::uint8_t> moved(geometry.vertices.size(), 0U);
        for (std::size_t v = 0; v < geometry.vertices.size(); ++v)
            for (std::size_t slot = 0; slot < 4U; ++slot) {
                const auto joint = static_cast<std::int32_t>(plain.joints[v][slot]);
                if (plain.weights[v][slot] > 0.0F && (joint == seg.start || under(plain.rig, joint, driver))) moved[v] = 1U;
            }
        float worst_plain = 1e9F, worst_twist = 1e9F;
        for (std::size_t i = 0; i < A; ++i) {
            const std::size_t q = s * A + i;
            std::vector<float> rp, rt;
            for (const auto & f : geometry.faces) {
                if (!moved[f[0]] && !moved[f[1]] && !moved[f[2]]) continue;
                float rest[9];
                for (int c = 0; c < 3; ++c) {
                    const vec3 & v = geometry.vertices[f[static_cast<std::size_t>(c)]];
                    rest[c * 3] = v.x; rest[c * 3 + 1] = v.y; rest[c * 3 + 2] = v.z;
                }
                const float a0 = area(rest, rest + 3, rest + 6);
                if (a0 < 1e-12F) continue;
                const auto at = [&](const std::vector<float> & x, std::uint32_t v) { return x.data() + (v * Q + q) * 3U; };
                rp.push_back(area(at(*p, f[0]), at(*p, f[1]), at(*p, f[2])) / a0);
                rt.push_back(area(at(*t, f[0]), at(*t, f[1]), at(*t, f[2])) / a0);
            }
            if (rp.empty()) continue;
            const auto pct = [](std::vector<float> & x) {
                const auto n = static_cast<std::ptrdiff_t>(x.size() / 100U);
                std::nth_element(x.begin(), x.begin() + n, x.end());
                return x[static_cast<std::size_t>(n)];
            };
            worst_plain = std::min(worst_plain, pct(rp));
            worst_twist = std::min(worst_twist, pct(rt));
        }
        checks.push_back({seg.start, worst_plain, worst_twist, worst_twist > worst_plain});
    }
    return checks;
}

} // namespace

result<twist_rig> add_twist_joints(const mesh & geometry, const skin & binding, const runtime_options & runtime, bool check) {
    std::array<bool, role_count> all{};
    all.fill(true);
    std::vector<std::size_t> role_of;
    auto first = build_twist(geometry, binding, runtime, all, role_of);
    if (!first || first->segments.empty() || !check) return first;
    auto checks = check_segments(geometry, binding, *first, runtime);
    if (!checks) return std::unexpected(checks.error());
    std::array<bool, role_count> kept{};
    bool every = true;
    for (std::size_t s = 0; s < checks->size(); ++s) {
        kept[role_of[s]] = (*checks)[s].kept;
        every = every && (*checks)[s].kept;
    }
    if (every) { first->checks = std::move(*checks); return first; }
    std::vector<std::size_t> second_roles;
    auto second = build_twist(geometry, binding, runtime, kept, second_roles);
    if (!second) return second;
    second->checks = std::move(*checks);
    return second;
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
