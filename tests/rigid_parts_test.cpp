// Rigid assemblies: a model made of separate closed solids (an anatomical
// skeleton) gets one joint per part; a character with open sheets or one big
// body keeps its soft weights. Each check runs next to a control that must
// show the defect. Plain checks (not assert) so a Release build keeps them.
#include "binding.hpp"
#include "rigid_parts.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using skintokens::triangle;
using skintokens::vec3;
namespace rigid = skintokens::detail::rigid;
using welded_face = std::array<std::uint32_t, 3>;

int failures = 0;

void check(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

struct model {
    std::vector<vec3> vertices;
    std::vector<welded_face> faces;
};

// A closed box split into `segments` rings along x (a "long bone"):
// (segments + 1) rings of 4 vertices, side quads, and two end caps.
void add_bone(model &m, vec3 low, vec3 high, int segments) {
    const auto base = static_cast<std::uint32_t>(m.vertices.size());
    for (int s = 0; s <= segments; ++s) {
        const float x = low.x + (high.x - low.x) * static_cast<float>(s) / static_cast<float>(segments);
        m.vertices.push_back({x, low.y, low.z});
        m.vertices.push_back({x, high.y, low.z});
        m.vertices.push_back({x, high.y, high.z});
        m.vertices.push_back({x, low.y, high.z});
    }
    for (int s = 0; s < segments; ++s)
        for (std::uint32_t k = 0; k < 4U; ++k) {
            const auto a = base + static_cast<std::uint32_t>(s) * 4U + k;
            const auto b = base + static_cast<std::uint32_t>(s) * 4U + (k + 1U) % 4U;
            m.faces.push_back({a, b, b + 4U});
            m.faces.push_back({a, b + 4U, a + 4U});
        }
    const auto end = base + static_cast<std::uint32_t>(segments) * 4U;
    m.faces.push_back({base, base + 2U, base + 1U});
    m.faces.push_back({base, base + 3U, base + 2U});
    m.faces.push_back({end, end + 1U, end + 2U});
    m.faces.push_back({end, end + 2U, end + 3U});
}

// An open sheet in the plane z = 0 (a hair card or a cloth panel).
void add_sheet(model &m, float x0, float y0, float size, int cells) {
    const auto base = static_cast<std::uint32_t>(m.vertices.size());
    const int side = cells + 1;
    for (int j = 0; j <= cells; ++j)
        for (int i = 0; i <= cells; ++i)
            m.vertices.push_back({x0 + size * static_cast<float>(i) / static_cast<float>(cells),
                y0 + size * static_cast<float>(j) / static_cast<float>(cells), 0.0F});
    for (int j = 0; j < cells; ++j)
        for (int i = 0; i < cells; ++i) {
            const auto a = base + static_cast<std::uint32_t>(j * side + i);
            m.faces.push_back({a, a + 1U, a + static_cast<std::uint32_t>(side) + 1U});
            m.faces.push_back({a, a + static_cast<std::uint32_t>(side) + 1U, a + static_cast<std::uint32_t>(side)});
        }
}

// Ten bones in a row along x, bone b spans [b, b + 0.8]. Joint b sits at the
// bone's start; the soft weights blend joints b and b + 1 linearly along the
// bone (the "bubble gum" the learned weights give), 75/25 at the far end.
constexpr int bones = 10;
constexpr std::size_t joints = bones + 1;

model skeleton(bool outline_on_bone_0) {
    model m;
    for (int b = 0; b < bones; ++b)
        add_bone(m, {static_cast<float>(b), 0.0F, 0.0F}, {static_cast<float>(b) + 0.8F, 0.2F, 0.2F}, 6);
    // The inverted-hull outline of bone 0: same topology, pushed out by 0.02.
    if (outline_on_bone_0) add_bone(m, {-0.02F, -0.02F, -0.02F}, {0.82F, 0.22F, 0.22F}, 6);
    return m;
}

std::vector<float> soft_rows(const model &m, std::size_t outline_first) {
    std::vector<float> rows(m.vertices.size() * joints, 0.0F);
    for (std::size_t v = 0; v < m.vertices.size(); ++v) {
        const float x = std::clamp(m.vertices[v].x, 0.0F, static_cast<float>(bones) - 0.2F);
        const auto b = static_cast<std::size_t>(std::floor(x));
        const float t = std::clamp(x - static_cast<float>(b), 0.0F, 0.8F) / 0.8F;
        float *row = rows.data() + v * joints;
        row[b] = 1.0F - 0.25F * t;
        row[b + 1U] = 0.25F * t;
        // The outline's own weights lean to joint 1 (60%): alone it would
        // pick a different joint than its bone and slide off it.
        if (v >= outline_first) {
            row[0] = 0.4F;
            row[1] = 0.6F;
        }
    }
    return rows;
}

float strongest_share_min(const model &m, const std::vector<float> &rows, std::size_t first, std::size_t last) {
    float worst = 1.0F;
    for (std::size_t v = first; v < last && v < m.vertices.size(); ++v) {
        const float *row = rows.data() + v * joints;
        worst = std::min(worst, *std::max_element(row, row + joints));
    }
    return worst;
}

std::size_t strongest(const std::vector<float> &rows, std::size_t v) {
    const float *row = rows.data() + v * joints;
    return static_cast<std::size_t>(std::max_element(row, row + joints) - row);
}

void parts_and_closedness() {
    model m = skeleton(false);
    add_sheet(m, 0.0F, 2.0F, 1.0F, 4);
    const auto found = rigid::find_parts(m.vertices.size(), m.faces);
    check(found.members.size() == static_cast<std::size_t>(bones) + 1U, "parts: ten bones and one sheet");
    std::size_t closed = 0U;
    for (const auto c : found.closed) closed += c ? 1U : 0U;
    check(closed == static_cast<std::size_t>(bones), "parts: the bones are closed, the sheet is open");
}

void assembly_decision() {
    const model bonesonly = skeleton(true);
    check(rigid::is_assembly(rigid::find_parts(bonesonly.vertices.size(), bonesonly.faces)),
        "assembly: separate closed bones are an assembly");

    // Mint-like: the same bones plus open hair/cloth sheets holding most of
    // the vertices. Nothing may change.
    model dressed = skeleton(true);
    for (int k = 0; k < 4; ++k) add_sheet(dressed, static_cast<float>(k), 3.0F, 1.0F, 8);
    check(!rigid::is_assembly(rigid::find_parts(dressed.vertices.size(), dressed.faces)),
        "assembly: a model with open sheets is not");

    // One closed body (a low-poly character) is not an assembly either.
    model body;
    add_bone(body, {0.0F, 0.0F, 0.0F}, {10.0F, 0.2F, 0.2F}, 60);
    for (int b = 0; b < bones; ++b)
        add_bone(body, {static_cast<float>(b), 1.0F, 0.0F}, {static_cast<float>(b) + 0.1F, 1.1F, 0.1F}, 1);
    check(!rigid::is_assembly(rigid::find_parts(body.vertices.size(), body.faces)),
        "assembly: one large closed body is not");
}

void snap_with_outline() {
    const model m = skeleton(true);
    const std::size_t outline_first = static_cast<std::size_t>(bones) * 28U;
    const auto found = rigid::find_parts(m.vertices.size(), m.faces);
    auto rows = soft_rows(m, outline_first);
    // Control: the soft rows bend (share 0.75 at the far end) and the
    // outline leans to joint 1 while its bone leans to joint 0.
    check(strongest_share_min(m, rows, 0U, outline_first) < 0.8F, "control: soft bones are blended");
    check(strongest(rows, outline_first) != strongest(rows, 0U), "control: alone, the outline picks another joint");

    // Without pairing (each part on its own) the outline slides off its bone.
    std::vector<std::uint32_t> alone(found.members.size());
    for (std::uint32_t p = 0; p < alone.size(); ++p) alone[p] = p;
    auto unpaired = rows;
    rigid::snap(found, alone, unpaired, joints);
    check(strongest(unpaired, outline_first) != strongest(unpaired, 0U), "control: unpaired outline differs from its bone");

    const auto group = rigid::pair_shells(found, m.vertices);
    std::size_t paired = 0U;
    for (std::uint32_t p = 0; p < group.size(); ++p) paired += group[p] != p ? 1U : 0U;
    check(paired == 1U, "pairing: exactly one part joins a shell (bone 0 with its outline)");
    const auto snapped = rigid::snap(found, group, rows, joints);
    check(snapped == found.members.size(), "snap: every part snapped");
    check(strongest_share_min(m, rows, 0U, m.vertices.size()) == 1.0F, "snap: every vertex follows one joint at 100%");
    check(strongest(rows, outline_first) == strongest(rows, 0U) && strongest(rows, 0U) == 0U,
        "snap: the outline follows its bone's joint");
    bool own_joint = true;
    for (int b = 0; b < bones; ++b) own_joint = own_joint && strongest(rows, static_cast<std::size_t>(b) * 28U) == static_cast<std::size_t>(b);
    check(own_joint, "snap: each bone keeps its strongest joint");
}

// The same skeleton through the real transfer: samples on the bone surfaces
// carry the soft rows. With rigid assemblies on, every vertex ends at one
// joint; with the option off (the control) bones stay blended.
void through_transfer() {
    const model m = skeleton(true);
    std::vector<vec3> normals(m.vertices.size(), vec3{0.0F, 0.0F, 1.0F});
    std::vector<triangle> faces(m.faces.begin(), m.faces.end());
    const auto rows = soft_rows(m, static_cast<std::size_t>(bones) * 28U);
    std::vector<vec3> points, point_normals;
    std::vector<std::uint32_t> point_faces;
    std::vector<std::vector<float>> dense(joints);
    for (std::size_t f = 0; f < faces.size(); ++f) {
        const auto &t = faces[f];
        const auto a = m.vertices[t[0]], b = m.vertices[t[1]], c = m.vertices[t[2]];
        const vec3 e1{b.x - a.x, b.y - a.y, b.z - a.z}, e2{c.x - a.x, c.y - a.y, c.z - a.z};
        vec3 n{e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x};
        const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        n = {n.x / len, n.y / len, n.z / len};
        for (const auto corner : t) {
            const auto p = m.vertices[corner];
            points.push_back({(2.0F * p.x + a.x + b.x + c.x - p.x) / 4.0F, (2.0F * p.y + a.y + b.y + c.y - p.y) / 4.0F,
                (2.0F * p.z + a.z + b.z + c.z - p.z) / 4.0F});
            point_normals.push_back(n);
            point_faces.push_back(static_cast<std::uint32_t>(f));
            for (std::size_t j = 0; j < joints; ++j) dense[j].push_back(rows[corner * joints + j]);
        }
    }
    skintokens::skeleton rig;
    for (std::size_t j = 0; j < joints; ++j) {
        rig.names.push_back("j" + std::to_string(j));
        rig.parents.push_back(static_cast<std::int32_t>(j) - 1);
        rig.rest_positions.push_back({static_cast<float>(j), 0.1F, 0.1F});
    }
    const auto run = [&](bool on) {
        skintokens::detail::robust_transfer_options options;
        options.rigid_assemblies = on;
        return skintokens::detail::integrate_robust_binding(rig, m.vertices, normals, faces, points, point_normals,
            point_faces, dense, options);
    };
    const auto top = [](const skintokens::skin &s, std::size_t v) { return s.weights[v][0]; };
    const auto soft = run(false);
    const auto hard = run(true);
    float soft_min = 1.0F, hard_min = 1.0F;
    for (std::size_t v = 0; v < m.vertices.size(); ++v) {
        soft_min = std::min(soft_min, top(soft, v));
        hard_min = std::min(hard_min, top(hard, v));
    }
    std::printf("     transfer: weakest top weight soft %.3f rigid %.3f, rigid parts %zu\n",
        static_cast<double>(soft_min), static_cast<double>(hard_min), hard.transfer.rigid_parts);
    check(soft_min < 0.9F && soft.transfer.rigid_parts == 0U, "control: transfer without rigid assemblies stays blended");
    check(hard_min == 1.0F, "transfer: every vertex of the assembly has one joint at 100%");
    check(hard.transfer.rigid_parts == static_cast<std::size_t>(bones) + 1U, "transfer: reports eleven rigid parts");
    const std::size_t outline_first = static_cast<std::size_t>(bones) * 28U;
    check(hard.joints[outline_first][0] == hard.joints[0][0], "transfer: the outline follows its bone");
}

} // namespace

int main() {
    parts_and_closedness();
    assembly_decision();
    snap_with_outline();
    through_transfer();
    std::printf("%s (%d failed)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
