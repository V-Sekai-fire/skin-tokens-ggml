// pose_to_t_pose_glb_memory puts a bent VRM 1.0 humanoid in a T-pose and binds it there;
// reframe_skin_joints_glb_memory turns each joint's +Y along its bone without moving anything.
// Each check first runs on the untouched input, where it must fail (the control).
#include <skintokens/skintokens.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;
using mat = std::array<float, 16>;  // column-major, as glTF
struct v3 { float x, y, z; };

template<class T> void append(std::vector<std::byte> & out, const T & value) {
    const auto offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

std::vector<std::byte> pack(std::string text, std::vector<std::byte> binary) {
    while (text.size() % 4U) text.push_back(' ');
    while (binary.size() % 4U) binary.push_back(std::byte{});
    std::vector<std::byte> out;
    append(out, 0x46546C67U); append(out, 2U);
    append(out, static_cast<std::uint32_t>(12U + 8U + text.size() + 8U + binary.size()));
    append(out, static_cast<std::uint32_t>(text.size())); append(out, 0x4E4F534AU);
    for (char c : text) out.push_back(static_cast<std::byte>(c));
    append(out, static_cast<std::uint32_t>(binary.size())); append(out, 0x004E4942U);
    out.insert(out.end(), binary.begin(), binary.end());
    return out;
}

struct glb { json root; const std::byte * bin; };
glb open(const std::vector<std::byte> & bytes) {
    std::uint32_t size = 0;
    std::memcpy(&size, bytes.data() + 12, 4);
    const auto * begin = reinterpret_cast<const char *>(bytes.data() + 20);
    return {json::parse(begin, begin + size), bytes.data() + 20 + size + 8};
}

// Float accessor element i, component k (tight or strided).
float element(const glb & g, std::size_t index, std::size_t i, std::size_t k, std::size_t width) {
    const json & a = g.root["accessors"][index];
    const json & view = g.root["bufferViews"][a["bufferView"].get<std::size_t>()];
    const std::size_t stride = view.value("byteStride", width * 4U);
    float value = 0;
    std::memcpy(&value, g.bin + view.value("byteOffset", 0U) + a.value("byteOffset", 0U) + i * stride + k * 4U, 4);
    return value;
}

mat multiply(const mat & a, const mat & b) {
    mat out{};
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            for (int k = 0; k < 4; ++k) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    return out;
}

mat local(const json & n) {
    const auto t = n.value("translation", std::vector<float>{0, 0, 0});
    const auto q = n.value("rotation", std::vector<float>{0, 0, 0, 1});
    const auto s = n.value("scale", std::vector<float>{1, 1, 1});
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    return {(1 - 2*(y*y + z*z)) * s[0], 2*(x*y + z*w) * s[0], 2*(x*z - y*w) * s[0], 0,
            2*(x*y - z*w) * s[1], (1 - 2*(x*x + z*z)) * s[1], 2*(y*z + x*w) * s[1], 0,
            2*(x*z + y*w) * s[2], 2*(y*z - x*w) * s[2], (1 - 2*(x*x + y*y)) * s[2], 0, t[0], t[1], t[2], 1};
}

std::vector<mat> worlds(const json & root) {
    const auto & nodes = root["nodes"];
    std::vector<int> parent(nodes.size(), -1);
    for (std::size_t n = 0; n < nodes.size(); ++n)
        for (const auto & c : nodes[n].value("children", json::array())) parent[c.get<std::size_t>()] = static_cast<int>(n);
    std::vector<mat> out(nodes.size());
    for (std::size_t n = 0; n < nodes.size(); ++n) {
        mat m = local(nodes[n]);
        for (int p = parent[n]; p >= 0; p = parent[static_cast<std::size_t>(p)]) m = multiply(local(nodes[static_cast<std::size_t>(p)]), m);
        out[n] = m;
    }
    return out;
}

v3 apply(const mat & m, v3 p) {
    return {m[0]*p.x + m[4]*p.y + m[8]*p.z + m[12], m[1]*p.x + m[5]*p.y + m[9]*p.z + m[13],
            m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14]};
}
v3 origin(const mat & m) { return {m[12], m[13], m[14]}; }
float distance(v3 a, v3 b) { return std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)); }
float cosine(v3 a, v3 b) {
    return (a.x*b.x + a.y*b.y + a.z*b.z) / std::sqrt((a.x*a.x + a.y*a.y + a.z*a.z) * (b.x*b.x + b.y*b.y + b.z*b.z));
}

// Joint names, parents and world rest positions of a humanoid with its arms down at 45 degrees
// and both knees bent: an arbitrary pose, not a T-pose.
const char * const names[] = {"hips", "spine", "head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm",
    "rightLowerArm", "rightHand", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"};
const int parents[] = {-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13};
const v3 rest[] = {{0, 1, 0}, {0, 1.2F, 0}, {0, 1.6F, 0}, {0.15F, 1.45F, 0}, {0.35F, 1.25F, 0.02F}, {0.5F, 1.05F, 0.1F},
    {-0.15F, 1.45F, 0}, {-0.33F, 1.24F, -0.03F}, {-0.47F, 1.06F, 0.05F}, {0.1F, 0.95F, 0}, {0.1F, 0.5F, 0.1F},
    {0.1F, 0.08F, 0}, {-0.1F, 0.95F, 0}, {-0.12F, 0.52F, 0.15F}, {-0.1F, 0.08F, 0.02F}};
constexpr std::size_t joints = 15U;
const v3 offset{0.02F, 0.01F, 0.03F};  // each vertex sits this far from its joint

std::vector<std::byte> fixture() {
    std::vector<std::byte> bin;
    for (std::size_t j = 0; j < joints; ++j) { append(bin, rest[j].x + offset.x); append(bin, rest[j].y + offset.y); append(bin, rest[j].z + offset.z); }
    for (std::size_t j = 0; j < joints; ++j) { append(bin, static_cast<std::uint16_t>(j)); for (int k = 0; k < 3; ++k) append(bin, std::uint16_t{0}); }
    for (std::size_t j = 0; j < joints; ++j) { append(bin, 1.0F); for (int k = 0; k < 3; ++k) append(bin, 0.0F); }
    for (std::size_t j = 0; j < joints; ++j) append(bin, static_cast<std::uint16_t>(j));
    while (bin.size() % 4U) bin.push_back(std::byte{});
    const std::size_t binds_at = bin.size();
    for (std::size_t j = 0; j < joints; ++j) {
        const mat inverse{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -rest[j].x, -rest[j].y, -rest[j].z, 1};
        for (float v : inverse) append(bin, v);
    }
    json nodes = json::array();
    json bones = json::object();
    for (std::size_t j = 0; j < joints; ++j) {
        const v3 up = parents[j] < 0 ? v3{0, 0, 0} : rest[parents[j]];
        nodes.push_back({{"name", names[j]}, {"translation", {rest[j].x - up.x, rest[j].y - up.y, rest[j].z - up.z}}});
        bones[names[j]] = {{"node", j}};
    }
    for (std::size_t j = 0; j < joints; ++j)
        if (parents[j] >= 0) nodes[static_cast<std::size_t>(parents[j])]["children"].push_back(j);
    nodes.push_back({{"name", "Body"}, {"mesh", 0}, {"skin", 0}});
    json skin_joints = json::array();
    for (std::size_t j = 0; j < joints; ++j) skin_joints.push_back(j);
    const json root = {
        {"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", {{{"nodes", {0, joints}}}}},
        {"extensionsUsed", {"VRMC_vrm", "VRMC_springBone"}},
        {"extensions", {{"VRMC_vrm", {{"specVersion", "1.0"}, {"humanoid", {{"humanBones", bones}}},
                                      {"lookAt", {{"offsetFromHeadBone", {0, 0.06, 0.08}}}}}},
                        {"VRMC_springBone", {{"specVersion", "1.0"},
                            {"colliders", {{{"node", 5}, {"shape", {{"sphere", {{"offset", {0.05, 0.01, 0.02}}, {"radius", 0.03}}}}}}}}}}}},
        {"nodes", nodes},
        {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}, {"JOINTS_0", 1}, {"WEIGHTS_0", 2}}}, {"indices", 3}}}}}}},
        {"skins", {{{"joints", skin_joints}, {"inverseBindMatrices", 4}}}},
        {"accessors", {
            {{"bufferView", 0}, {"componentType", 5126}, {"count", joints}, {"type", "VEC3"}, {"min", {-1, 0, -1}}, {"max", {1, 2, 1}}},
            {{"bufferView", 1}, {"componentType", 5123}, {"count", joints}, {"type", "VEC4"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", joints}, {"type", "VEC4"}},
            {{"bufferView", 3}, {"componentType", 5123}, {"count", joints}, {"type", "SCALAR"}},
            {{"bufferView", 4}, {"componentType", 5126}, {"count", joints}, {"type", "MAT4"}}}},
        {"bufferViews", {
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", joints * 12U}},
            {{"buffer", 0}, {"byteOffset", joints * 12U}, {"byteLength", joints * 8U}},
            {{"buffer", 0}, {"byteOffset", joints * 20U}, {"byteLength", joints * 16U}},
            {{"buffer", 0}, {"byteOffset", joints * 36U}, {"byteLength", joints * 2U}},
            {{"buffer", 0}, {"byteOffset", binds_at}, {"byteLength", joints * 64U}}}},
        {"buffers", {{{"byteLength", bin.size()}}}}};
    return pack(root.dump(), bin);
}

// Every vertex where the skin puts it at rest (joint world times inverse bind times position).
std::vector<v3> skinned(const glb & g) {
    const auto w = worlds(g.root);
    const json & skin = g.root["skins"][0];
    const json & primitive = g.root["meshes"][0]["primitives"][0];
    const std::size_t positions = primitive["attributes"]["POSITION"].get<std::size_t>();
    const std::size_t binds = skin["inverseBindMatrices"].get<std::size_t>();
    std::vector<v3> out;
    for (std::size_t v = 0; v < joints; ++v) {
        mat inverse{};
        for (std::size_t k = 0; k < 16U; ++k) inverse[k] = element(g, binds, v, k, 16U);  // vertex v is bound to joint v
        const mat m = multiply(w[skin["joints"][v].get<std::size_t>()], inverse);
        out.push_back(apply(m, {element(g, positions, v, 0, 3), element(g, positions, v, 1, 3), element(g, positions, v, 2, 3)}));
    }
    return out;
}

// The worst angle (degrees) between a limb and its T-pose direction.
float t_pose_error(const glb & g) {
    const auto w = worlds(g.root);
    const struct { int from, to; v3 direction; } limbs[] = {
        {3, 4, {1, 0, 0}}, {4, 5, {1, 0, 0}}, {6, 7, {-1, 0, 0}}, {7, 8, {-1, 0, 0}},
        {9, 10, {0, -1, 0}}, {10, 11, {0, -1, 0}}, {12, 13, {0, -1, 0}}, {13, 14, {0, -1, 0}}};
    float worst = 0;
    for (const auto & limb : limbs) {
        const v3 a = origin(w[limb.from]), b = origin(w[limb.to]);
        const float c = cosine({b.x - a.x, b.y - a.y, b.z - a.z}, limb.direction);
        worst = std::fmax(worst, std::acos(std::fmin(1.0F, std::fmax(-1.0F, c))) * 57.29578F);
    }
    return worst;
}

// The worst angle (degrees) between a joint's +Y and its bone to the next humanoid joint.
float frame_error(const glb & g) {
    const auto w = worlds(g.root);
    const int next[] = {1, 2, -1, 4, 5, -1, 7, 8, -1, 10, 11, -1, 13, 14, -1};
    float worst = 0;
    for (std::size_t j = 0; j < joints; ++j) {
        if (next[j] < 0) continue;
        const v3 a = origin(w[j]), b = origin(w[static_cast<std::size_t>(next[j])]);
        const float c = cosine({w[j][4], w[j][5], w[j][6]}, {b.x - a.x, b.y - a.y, b.z - a.z});
        worst = std::fmax(worst, std::acos(std::fmin(1.0F, std::fmax(-1.0F, c))) * 57.29578F);
    }
    return worst;
}

v3 collider(const glb & g) {
    const auto & sphere = g.root["extensions"]["VRMC_springBone"]["colliders"][0]["shape"]["sphere"]["offset"];
    return apply(worlds(g.root)[5], {sphere[0].get<float>(), sphere[1].get<float>(), sphere[2].get<float>()});
}
v3 look_at(const glb & g) {
    const auto & o = g.root["extensions"]["VRMC_vrm"]["lookAt"]["offsetFromHeadBone"];
    return apply(worlds(g.root)[2], {o[0].get<float>(), o[1].get<float>(), o[2].get<float>()});
}

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)
} // namespace

int main() {
    const auto source = fixture();
    const glb input = open(source);

    // Controls: the bent input is far from a T-pose and its joints are not framed along their bones.
    CHECK(t_pose_error(input) > 30.0F);
    CHECK(frame_error(input) > 30.0F);
    const auto before = skinned(input);
    for (std::size_t v = 0; v < joints; ++v) CHECK(distance(before[v], {rest[v].x + offset.x, rest[v].y + offset.y, rest[v].z + offset.z}) < 1e-6F);

    // T-pose: every limb on its direction, the mesh bound at the new rest, each vertex the same
    // distance from its joint as before.
    auto posed_bytes = skintokens::pose_to_t_pose_glb_memory(source);
    if (!posed_bytes) { std::fprintf(stderr, "t-pose: %s\n", posed_bytes.error().message.c_str()); return 1; }
    const glb posed = open(*posed_bytes);
    CHECK(t_pose_error(posed) < 0.01F);
    const auto posed_mesh = skinned(posed);
    const auto posed_worlds = worlds(posed.root);
    const json & posed_positions = posed.root["meshes"][0]["primitives"][0]["attributes"]["POSITION"];
    for (std::size_t v = 0; v < joints; ++v) {
        const v3 stored{element(posed, posed_positions.get<std::size_t>(), v, 0, 3), element(posed, posed_positions.get<std::size_t>(), v, 1, 3),
                        element(posed, posed_positions.get<std::size_t>(), v, 2, 3)};
        CHECK(distance(posed_mesh[v], stored) < 1e-5F);  // bound where it is drawn
        CHECK(std::abs(distance(posed_mesh[v], origin(posed_worlds[v])) - distance(before[v], rest[v])) < 1e-5F);
    }
    CHECK(distance(origin(posed_worlds[0]), rest[0]) < 1e-6F && distance(origin(posed_worlds[2]), rest[2]) < 1e-6F);
    CHECK(distance(posed_mesh[5], before[5]) > 0.1F);  // the left hand moved up into the T

    // A file with clips keeps its rest pose: refused.
    {
        glb with_clip = open(source);
        with_clip.root["animations"] = json::array({{{"channels", json::array()}, {"samplers", json::array()}}});
        auto clip_source = pack(with_clip.root.dump(), std::vector<std::byte>(input.bin, source.data() + source.size()));
        CHECK(!skintokens::pose_to_t_pose_glb_memory(clip_source));
    }

    // Re-frame: each joint's +Y along its bone, and nothing moves.
    auto framed_bytes = skintokens::reframe_skin_joints_glb_memory(*posed_bytes);
    if (!framed_bytes) { std::fprintf(stderr, "reframe: %s\n", framed_bytes.error().message.c_str()); return 1; }
    const glb framed = open(*framed_bytes);
    CHECK(frame_error(posed) > 30.0F);  // control: the T-pose alone does not frame the joints
    CHECK(frame_error(framed) < 0.01F);
    CHECK(t_pose_error(framed) < 0.01F);
    const auto framed_mesh = skinned(framed);
    for (std::size_t v = 0; v < joints; ++v) CHECK(distance(framed_mesh[v], posed_mesh[v]) < 1e-5F);
    CHECK(distance(collider(framed), collider(posed)) < 1e-5F);
    CHECK(distance(look_at(framed), look_at(posed)) < 1e-5F);
    // The frame really turned: the left upper arm's +Y was world +Y and is now along +X.
    CHECK(worlds(framed.root)[3][4] > 0.9999F);

    // Not a VRM 1.0 humanoid: the T-pose is refused.
    {
        json plain = input.root;
        plain["extensions"].erase("VRMC_vrm");
        CHECK(!skintokens::pose_to_t_pose_glb_memory(pack(plain.dump(), std::vector<std::byte>(input.bin, source.data() + source.size()))));
    }
    std::puts("vrm-pose-test: PASS");
    return 0;
}
