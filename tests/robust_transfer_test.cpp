// Robust weight transfer: each check runs the upstream interpolation as the
// control, which must show the defect, next to the robust transfer, which must
// not. Plain checks (not assert) so a Release build cannot drop them.
#include "binding.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using skintokens::triangle;
using skintokens::vec3;

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

struct patch {
    std::vector<vec3> vertices, normals;
    std::vector<triangle> faces;
};

// A grid of rectangles in the plane z = 0, facing +z.
void add_rect(patch & out, float x0, float y0, float w, float h, int cx, int cy) {
    const auto base = static_cast<std::uint32_t>(out.vertices.size());
    const int side = cx + 1;
    for (int j = 0; j <= cy; ++j)
        for (int i = 0; i <= cx; ++i) {
            out.vertices.push_back({x0 + w * static_cast<float>(i) / static_cast<float>(cx),
                                    y0 + h * static_cast<float>(j) / static_cast<float>(cy), 0.0F});
            out.normals.push_back({0.0F, 0.0F, 1.0F});
        }
    for (int j = 0; j < cy; ++j)
        for (int i = 0; i < cx; ++i) {
            const auto a = base + static_cast<std::uint32_t>(j * side + i);
            const auto b = a + 1U, c = a + static_cast<std::uint32_t>(side), d = c + 1U;
            out.faces.push_back({a, b, d});
            out.faces.push_back({a, d, c});
        }
}

void add_grid(patch & out, float x0, float y0, float size, int cells) {
    add_rect(out, x0, y0, size, size, cells, cells);
}

// Samples at every face centre and three interior points per face, with the face
// normal and the face index; `label(point, face)` gives the learned joint.
template<class Label>
void sample(const patch & mesh, std::size_t joints, Label label, std::vector<vec3> & points,
            std::vector<vec3> & normals, std::vector<std::uint32_t> & faces,
            std::vector<std::vector<float>> & dense) {
    dense.assign(joints, {});
    const auto add = [&](vec3 p, std::uint32_t face) {
        points.push_back(p);
        normals.push_back({0.0F, 0.0F, 1.0F});
        faces.push_back(face);
        const auto joint = label(p, face);
        for (std::size_t j = 0; j < joints; ++j) dense[j].push_back(j == joint ? 1.0F : 0.0F);
    };
    for (std::uint32_t f = 0; f < mesh.faces.size(); ++f) {
        const auto & t = mesh.faces[f];
        const auto a = mesh.vertices[t[0]], b = mesh.vertices[t[1]], c = mesh.vertices[t[2]];
        add({(a.x + b.x + c.x) / 3.0F, (a.y + b.y + c.y) / 3.0F, 0.0F}, f);
        const auto mix = [](vec3 p, vec3 q, vec3 r) {
            return vec3{(4.0F * p.x + q.x + r.x) / 6.0F, (4.0F * p.y + q.y + r.y) / 6.0F, 0.0F};
        };
        add(mix(a, b, c), f);
        add(mix(b, c, a), f);
        add(mix(c, a, b), f);
    }
}

skintokens::skeleton rig(std::size_t joints) {
    skintokens::skeleton out;
    for (std::size_t j = 0; j < joints; ++j) {
        out.names.push_back("j" + std::to_string(j));
        out.parents.push_back(j == 0U ? -1 : 0);
        out.rest_positions.push_back({static_cast<float>(j), 0.0F, 0.0F});
    }
    return out;
}

float weight_of(const skintokens::skin & s, std::size_t vertex, std::uint16_t joint) {
    float total = 0.0F;
    for (std::size_t slot = 0; slot < 4U; ++slot)
        if (s.joints[vertex][slot] == joint) total += s.weights[vertex][slot];
    return total;
}

} // namespace

int main() {
    using skintokens::detail::integrate_learned_binding;
    using skintokens::detail::integrate_robust_binding;

    // 1. Two legs side by side: separate parts 0.02 apart. Plain
    // interpolation pulls the right leg's joint into the left leg's inner edge.
    {
        patch legs;
        add_grid(legs, 0.0F, 0.0F, 1.0F, 10);
        add_grid(legs, 1.02F, 0.0F, 1.0F, 10);
        const std::size_t left = 121U;
        std::vector<vec3> points, normals;
        std::vector<std::uint32_t> faces;
        std::vector<std::vector<float>> dense;
        sample(legs, 2U, [&](vec3, std::uint32_t face) { return face < 200U ? 0U : 1U; }, points, normals, faces, dense);
        const auto target = rig(2U);
        const auto raw = integrate_learned_binding(target, legs.vertices, points, dense);
        const auto robust = integrate_robust_binding(target, legs.vertices, legs.normals, legs.faces,
                                                     points, normals, faces, dense);
        float raw_leak = 0.0F, robust_leak = 0.0F;
        for (std::size_t v = 0; v < legs.vertices.size(); ++v) {
            const auto wrong = static_cast<std::uint16_t>(v < left ? 1U : 0U);
            raw_leak = std::max(raw_leak, weight_of(raw, v, wrong));
            robust_leak = std::max(robust_leak, weight_of(robust, v, wrong));
        }
        std::printf("     legs: raw leak %.3f, robust leak %.3f, matched %zu of %zu\n",
            static_cast<double>(raw_leak), static_cast<double>(robust_leak), robust.transfer.matched, robust.transfer.welded);
        check(raw_leak > 0.1F, "control: plain interpolation leaks across the gap between parts");
        check(robust_leak < 1.0e-6F, "robust: no vertex takes the other part's joint");
    }

    // 2. One grid, joint 0 everywhere except a cluster of samples on the
    // centre vertex labelled joint 1 (a decoder spike).
    {
        patch sheet;
        add_grid(sheet, 0.0F, 0.0F, 1.0F, 10);
        const std::size_t centre = 5U * 11U + 5U;
        const auto c = sheet.vertices[centre];
        std::vector<vec3> points, normals;
        std::vector<std::uint32_t> faces;
        std::vector<std::vector<float>> dense;
        sample(sheet, 2U, [&](vec3 p, std::uint32_t) {
            return std::hypot(p.x - c.x, p.y - c.y) < 0.045F ? 1U : 0U;
        }, points, normals, faces, dense);
        const auto target = rig(2U);
        const auto raw = integrate_learned_binding(target, sheet.vertices, points, dense);
        const auto robust = integrate_robust_binding(target, sheet.vertices, sheet.normals, sheet.faces,
                                                     points, normals, faces, dense);
        std::printf("     spike: raw %.3f, robust %.3f on joint 1, outliers %zu, inpainted %zu, iterations %zu\n",
            static_cast<double>(weight_of(raw, centre, 1U)), static_cast<double>(weight_of(robust, centre, 1U)),
            robust.transfer.outliers, robust.transfer.inpainted, robust.transfer.iterations);
        check(weight_of(raw, centre, 1U) > 0.5F, "control: plain interpolation binds the spike vertex to joint 1");
        check(robust.transfer.outliers >= 1U, "robust: the spike vertex is found as an outlier");
        check(weight_of(robust, centre, 0U) > 0.5F, "robust: the spike vertex is inpainted to joint 0");
        skintokens::detail::robust_transfer_options no_outliers;
        no_outliers.outlier_l1 = 1.0e9F;
        no_outliers.support_share = 0.0F;
        no_outliers.smooth_iterations = 0U;
        const auto planted = integrate_robust_binding(target, sheet.vertices, sheet.normals, sheet.faces,
                                                      points, normals, faces, dense, no_outliers);
        check(weight_of(planted, centre, 1U) > 0.5F, "plant: with the outlier test off the spike stays");
    }

    // 2b. Two fingers on one hand: one connected mesh (the base welds to both
    // prongs), prongs 0.02 apart, same facing. Only the surface neighbourhood
    // tells the fingers apart.
    {
        patch hand;
        add_rect(hand, 0.0F, 0.2F, 0.45F, 0.8F, 9, 16);   // left finger, faces 0-287
        add_rect(hand, 0.47F, 0.2F, 0.45F, 0.8F, 9, 16);  // right finger, faces 288-575
        add_rect(hand, 0.0F, 0.0F, 0.45F, 0.2F, 9, 4);
        add_rect(hand, 0.45F, 0.0F, 0.02F, 0.2F, 1, 4);
        add_rect(hand, 0.47F, 0.0F, 0.45F, 0.2F, 9, 4);
        std::vector<vec3> points, normals;
        std::vector<std::uint32_t> faces;
        std::vector<std::vector<float>> dense;
        sample(hand, 3U, [](vec3, std::uint32_t face) { return face < 288U ? 1U : face < 576U ? 2U : 0U; },
               points, normals, faces, dense);
        const auto target = rig(3U);
        const auto raw = integrate_learned_binding(target, hand.vertices, points, dense);
        const auto robust = integrate_robust_binding(target, hand.vertices, hand.normals, hand.faces,
                                                     points, normals, faces, dense);
        float raw_leak = 0.0F, robust_leak = 0.0F;
        for (std::size_t v = 0; v < hand.vertices.size(); ++v) {
            const auto p = hand.vertices[v];
            // Far from the base: smoothing may carry weight along the surface
            // through the base, never across the gap.
            if (p.y < 0.6F) continue;
            const auto wrong = static_cast<std::uint16_t>(p.x < 0.46F ? 2U : 1U);
            raw_leak = std::max(raw_leak, weight_of(raw, v, wrong));
            robust_leak = std::max(robust_leak, weight_of(robust, v, wrong));
        }
        std::printf("     fingers: raw leak %.3f, robust leak %.3f, matched %zu of %zu, inpainted %zu\n",
            static_cast<double>(raw_leak), static_cast<double>(robust_leak), robust.transfer.matched,
            robust.transfer.welded, robust.transfer.inpainted);
        check(raw_leak > 0.1F, "control: plain interpolation leaks between neighbouring fingers");
        check(robust_leak < 1.0e-3F, "robust: no finger vertex takes the other finger's joint");
        skintokens::detail::robust_transfer_options whole_part;
        whole_part.geodesic_ratio = 1000.0F;
        whole_part.smooth_iterations = 0U;
        const auto planted = integrate_robust_binding(target, hand.vertices, hand.normals, hand.faces,
                                                      points, normals, faces, dense, whole_part);
        float planted_leak = 0.0F;
        for (std::size_t v = 0; v < hand.vertices.size(); ++v) {
            const auto p = hand.vertices[v];
            if (p.y < 0.6F) continue;
            planted_leak = std::max(planted_leak, weight_of(planted, v, static_cast<std::uint16_t>(p.x < 0.46F ? 2U : 1U)));
        }
        check(planted_leak > 0.1F, "plant: with the neighbourhood grown to the whole part the fingers leak");
    }

    // 3. A seam: the sheet's middle column is duplicated (split UVs), the
    // duplicates get different, wrong normals. Twins must share one row.
    {
        patch sheet;
        add_grid(sheet, 0.0F, 0.0F, 1.0F, 4);
        const auto twin = static_cast<std::uint32_t>(sheet.vertices.size());
        sheet.vertices.push_back(sheet.vertices[12]);
        sheet.normals.push_back({0.0F, 0.0F, -1.0F});
        for (auto & f : sheet.faces)
            for (auto & corner : f)
                if (corner == 12U && (&f - sheet.faces.data()) % 2 == 0) corner = twin;
        std::vector<vec3> points, normals;
        std::vector<std::uint32_t> faces;
        std::vector<std::vector<float>> dense;
        sample(sheet, 3U, [&](vec3 p, std::uint32_t) { return p.x < 0.5F ? 0U : (p.y < 0.5F ? 1U : 2U); },
               points, normals, faces, dense);
        const auto robust = integrate_robust_binding(rig(3U), sheet.vertices, sheet.normals, sheet.faces,
                                                     points, normals, faces, dense);
        bool same = robust.joints[12] == robust.joints[twin];
        for (std::size_t slot = 0; slot < 4U; ++slot)
            same = same && std::abs(robust.weights[12][slot] - robust.weights[twin][slot]) < 1.0e-7F;
        check(same, "robust: seam twins share one weight row");
        float sum = 0.0F;
        for (const auto w : robust.weights[twin]) sum += w;
        check(std::abs(sum - 1.0F) < 1.0e-5F, "robust: weights sum to one");
    }

    // 4. Two shoes, two separate parts. Skeleton: 0 hips; 1-2 right leg and
    // foot; 3-4 left leg and foot. The right shoe's decoded weights carry
    // 30% left foot; the left shoe is clean. A third, connected body part
    // carries every joint except the bare hips (one empty joint in a row may
    // be crossed) and must keep them all.
    {
        patch scene;
        add_grid(scene, 0.0F, 0.0F, 0.3F, 3);    // right shoe, faces 0-17
        add_grid(scene, 0.5F, 0.0F, 0.3F, 3);    // left shoe, faces 18-35
        add_grid(scene, 0.0F, 1.0F, 1.0F, 5);    // body, faces 36-85
        std::vector<vec3> points, normals;
        std::vector<std::uint32_t> faces;
        std::vector<std::vector<float>> dense;
        sample(scene, 5U, [](vec3, std::uint32_t) { return 0U; }, points, normals, faces, dense);
        for (std::size_t i = 0; i < points.size(); ++i) {
            for (auto & column : dense) column[i] = 0.0F;
            if (faces[i] < 18U) { dense[2][i] = 0.7F; dense[4][i] = 0.3F; }
            else if (faces[i] < 36U) { dense[4][i] = 1.0F; }
            else { for (std::size_t j = 1; j < 5U; ++j) dense[j][i] = 0.25F; }  // bare hips joint 0
        }
        skintokens::skeleton legs;
        const std::int32_t parents[]{-1, 0, 1, 0, 3};
        for (std::size_t j = 0; j < 5U; ++j) {
            legs.names.push_back("j" + std::to_string(j));
            legs.parents.push_back(parents[j]);
            legs.rest_positions.push_back({static_cast<float>(j), 0.0F, 0.0F});
        }
        const auto robust = integrate_robust_binding(legs, scene.vertices, scene.normals, scene.faces,
                                                     points, normals, faces, dense);
        skintokens::detail::robust_transfer_options off;
        off.support_share = 0.0F;
        const auto planted = integrate_robust_binding(legs, scene.vertices, scene.normals, scene.faces,
                                                      points, normals, faces, dense, off);
        float right_shoe_left_foot = 0.0F, planted_left_foot = 0.0F, body_min = 1.0F;
        for (std::size_t v = 0; v < 16U; ++v) {
            right_shoe_left_foot = std::max(right_shoe_left_foot, weight_of(robust, v, 4U));
            planted_left_foot = std::max(planted_left_foot, weight_of(planted, v, 4U));
        }
        for (std::size_t v = 32U; v < scene.vertices.size(); ++v)
            for (std::uint16_t j = 1U; j < 5U; ++j) body_min = std::min(body_min, weight_of(robust, v, j));
        std::printf("     shoes: right shoe on left foot %.3f (support off %.3f), trimmed parts %zu, body min %.3f\n",
            static_cast<double>(right_shoe_left_foot), static_cast<double>(planted_left_foot),
            robust.transfer.trimmed_parts, static_cast<double>(body_min));
        check(planted_left_foot > 0.25F, "plant: with part support off the right shoe keeps the left foot");
        check(right_shoe_left_foot == 0.0F, "robust: the right shoe drops the left foot");
        check(robust.transfer.trimmed_parts == 1U, "robust: only the right shoe is trimmed");
        check(body_min > 0.1F, "robust: a connected body keeps every joint it carries");
    }

    // 4b. A whole body as one part: the fingertip (joint 3, 16 vertices) under
    // the hand (joint 2) carries a few percent of the vertices; both stay.
    {
        patch body;
        add_grid(body, 0.0F, 0.0F, 1.0F, 20);   // 441 vertices, 800 faces
        std::vector<vec3> points, normals;
        std::vector<std::uint32_t> faces;
        std::vector<std::vector<float>> dense;
        sample(body, 4U, [](vec3 p, std::uint32_t) {
            return p.x > 0.84F && p.y > 0.84F ? 3U : p.x > 0.69F && p.y > 0.69F ? 2U : p.x < 0.5F ? 0U : 1U;
        }, points, normals, faces, dense);
        skintokens::skeleton arm;
        const std::int32_t parents[]{-1, 0, 1, 2};
        for (std::size_t j = 0; j < 4U; ++j) {
            arm.names.push_back("j" + std::to_string(j));
            arm.parents.push_back(parents[j]);
            arm.rest_positions.push_back({static_cast<float>(j), 0.0F, 0.0F});
        }
        const auto robust = integrate_robust_binding(arm, body.vertices, body.normals, body.faces,
                                                     points, normals, faces, dense);
        const std::size_t tip = body.vertices.size() - 1U;  // (1, 1)
        std::printf("     fingertip: tip weight %.3f on joint 3, trimmed parts %zu\n",
            static_cast<double>(weight_of(robust, tip, 3U)), robust.transfer.trimmed_parts);
        check(weight_of(robust, tip, 3U) > 0.5F, "robust: a fingertip on a whole-body part keeps its joint");
        skintokens::detail::robust_transfer_options coarse;
        coarse.support_share = 0.10F;
        const auto planted = integrate_robust_binding(arm, body.vertices, body.normals, body.faces,
                                                      points, normals, faces, dense, coarse);
        check(robust.transfer.trimmed_parts == 0U, "robust: the whole body is not trimmed");
        check(planted.transfer.trimmed_parts == 1U, "plant: with a 10% share floor the body is trimmed");
    }

    // 5. Smoothing: a sheet split down the middle between joints 0 and 1
    // with one stray vertex in the joint-0 half bound fully to joint 1 by a
    // matched sample cluster too wide for the outlier test. Smoothing must
    // pull it towards its ring (control: smoothing off keeps it).
    {
        patch sheet;
        add_grid(sheet, 0.0F, 0.0F, 1.0F, 10);
        const std::size_t stray = 5U * 11U + 2U;  // (0.2, 0.5)
        const auto c = sheet.vertices[stray];
        std::vector<vec3> points, normals;
        std::vector<std::uint32_t> faces;
        std::vector<std::vector<float>> dense;
        sample(sheet, 2U, [&](vec3 p, std::uint32_t) {
            return p.x >= 0.5F || std::hypot(p.x - c.x, p.y - c.y) < 0.045F ? 1U : 0U;
        }, points, normals, faces, dense);
        skintokens::detail::robust_transfer_options no_outlier;
        no_outlier.outlier_l1 = 1.0e9F;
        auto unsmoothed = no_outlier;
        unsmoothed.smooth_iterations = 0U;
        const auto smoothed = integrate_robust_binding(rig(2U), sheet.vertices, sheet.normals, sheet.faces,
                                                       points, normals, faces, dense, no_outlier);
        const auto planted = integrate_robust_binding(rig(2U), sheet.vertices, sheet.normals, sheet.faces,
                                                      points, normals, faces, dense, unsmoothed);
        std::printf("     smoothing: stray vertex on joint 1: %.3f smoothed, %.3f unsmoothed\n",
            static_cast<double>(weight_of(smoothed, stray, 1U)), static_cast<double>(weight_of(planted, stray, 1U)));
        check(weight_of(planted, stray, 1U) > 0.5F, "plant: without smoothing the stray vertex stays on joint 1");
        check(weight_of(smoothed, stray, 1U) < 0.5F, "robust: smoothing pulls the stray vertex to its ring");
    }

    std::printf("%s (%d failed)\n", failures == 0 ? "ROBUST-TRANSFER-TEST PASS" : "ROBUST-TRANSFER-TEST FAIL", failures);
    return failures == 0 ? 0 : 1;
}
