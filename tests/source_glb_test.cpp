// save_rigged_source_glb_memory keeps the source GLB's materials, textures and UVs, and the old
// writer (the control) does not.
#include <skintokens/skintokens.hpp>
#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;

template<class T> void append(std::vector<std::byte> & output, const T & value) {
    const auto offset = output.size();
    output.resize(offset + sizeof(T));
    std::memcpy(output.data() + offset, &value, sizeof(T));
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

json read_json(const std::vector<std::byte> & glb) {
    std::uint32_t size = 0;
    std::memcpy(&size, glb.data() + 12, 4);
    const auto * begin = reinterpret_cast<const char *>(glb.data() + 20);
    return json::parse(begin, begin + size);
}

// Mesh 0: two triangles, two materials, NORMAL + TEXCOORD_0, placed twice (mirrored and plain).
// Mesh 1: a triangle without indices, which the rig skips. One embedded image.
std::vector<std::byte> fixture() {
    const float positions[] = {0,0,0, 1,0,0, 0,1,0,  0,0,1, 1,0,1, 0,1,1};
    const float normals[] = {1,0,0, 1,0,0, 1,0,0,  0,0,1, 0,0,1, 0,0,1};
    const float uvs[] = {0,0, 1,0, 0,1,  0,0, 1,0, 0,1};
    const std::uint16_t indices[] = {0,1,2};
    std::vector<std::byte> bin;
    for (float v : positions) append(bin, v);   // 0..72
    for (float v : normals) append(bin, v);     // 72..144
    for (float v : uvs) append(bin, v);         // 144..192
    for (auto v : indices) append(bin, v);      // 192..198
    while (bin.size() % 4U) bin.push_back(std::byte{});
    for (int i = 0; i < 8; ++i) bin.push_back(std::byte{0x89}); // 200..208 image bytes
    const std::string text = R"({"asset":{"version":"2.0","generator":"fixture"},"scene":0,
"scenes":[{"nodes":[0,1,2]}],
"nodes":[{"name":"Left","mesh":0,"translation":[1,0,0],"scale":[-1,1,1]},{"name":"Right","mesh":0},{"name":"Loose","mesh":1}],
"meshes":[{"name":"Body","primitives":[
 {"attributes":{"POSITION":0,"NORMAL":2,"TEXCOORD_0":4},"indices":6,"material":0},
 {"attributes":{"POSITION":1,"NORMAL":3,"TEXCOORD_0":5},"indices":6,"material":1}]},
 {"name":"Unindexed","primitives":[{"attributes":{"POSITION":0},"material":1}]}],
"materials":[{"name":"Skin","pbrMetallicRoughness":{"baseColorTexture":{"index":0}}},{"name":"Cloth","pbrMetallicRoughness":{"baseColorFactor":[0.2,0.4,0.6,1]}}],
"textures":[{"source":0}],"images":[{"bufferView":4,"mimeType":"image/png"}],
"accessors":[
 {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
 {"bufferView":0,"byteOffset":36,"componentType":5126,"count":3,"type":"VEC3"},
 {"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},
 {"bufferView":1,"byteOffset":36,"componentType":5126,"count":3,"type":"VEC3"},
 {"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"},
 {"bufferView":2,"byteOffset":24,"componentType":5126,"count":3,"type":"VEC2"},
 {"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}],
"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":72},{"buffer":0,"byteOffset":72,"byteLength":72},
 {"buffer":0,"byteOffset":144,"byteLength":48},{"buffer":0,"byteOffset":192,"byteLength":6},
 {"buffer":0,"byteOffset":200,"byteLength":8}],
"buffers":[{"byteLength":208}]})";
    return pack(text, bin);
}

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)
} // namespace

int main() {
    const auto source = fixture();
    auto geometry = skintokens::load_glb_memory(source);
    CHECK(geometry && geometry->vertices.size() == 12U && geometry->faces.size() == 4U);

    skintokens::skin binding;
    binding.rig.names = {"Hips", "Spine"};
    binding.rig.parents = {-1, 0};
    binding.rig.rest_positions = {{0, 0, 0}, {0, 1, 0}};
    binding.joints.assign(geometry->vertices.size(), {0, 1, 0, 0});
    binding.weights.assign(geometry->vertices.size(), {0.75F, 0.25F, 0.0F, 0.0F});
    skintokens::motion rest;
    rest.frames = 1U;
    rest.rig = binding.rig;
    rest.root_translations.assign(1U, binding.rig.rest_positions.front());
    rest.local_rotations.assign(binding.rig.names.size(), {});

    auto out = skintokens::save_rigged_source_glb_memory(source, binding, rest);
    if (!out) { std::fprintf(stderr, "writer: %s\n", out.error().message.c_str()); return 1; }
    const json root = read_json(*out);

    // Materials, textures and images stay; every skinned primitive keeps its material and UVs.
    CHECK(root["materials"].size() == 2U && root["materials"][1]["name"] == "Cloth");
    CHECK(root["textures"].size() == 1U && root["images"].size() == 1U);
    CHECK(root["asset"]["generator"] == "fixture");
    std::size_t skinned = 0;
    for (const auto & node : root["nodes"]) {
        if (!node.contains("skin")) continue;
        ++skinned;
        CHECK(node["skin"] == 0);
        const auto & primitives = root["meshes"][node["mesh"].get<std::size_t>()]["primitives"];
        CHECK(primitives.size() == 2U);
        CHECK(primitives[0]["material"] == 0 && primitives[1]["material"] == 1);
        for (const auto & primitive : primitives) {
            CHECK(primitive["attributes"].contains("TEXCOORD_0"));
            CHECK(primitive["attributes"].contains("JOINTS_0") && primitive["attributes"].contains("WEIGHTS_0"));
        }
    }
    CHECK(skinned == 2U);
    CHECK(!root["nodes"][0].contains("mesh") && !root["nodes"][1].contains("mesh"));
    CHECK(root["nodes"][2]["mesh"] == 1);  // the skipped primitive stays where it was
    CHECK(root["animations"].size() == 1U && root["animations"][0]["name"] == "RESET");
    CHECK(root["skins"][0]["joints"].size() == 2U);

    // Reading the result back gives the same world-space mesh the rig saw.
    auto again = skintokens::load_glb_memory(*out);
    CHECK(again && again->vertices.size() == geometry->vertices.size() && again->faces.size() == geometry->faces.size());
    for (std::size_t i = 0; i < geometry->vertices.size(); ++i) {
        CHECK(again->vertices[i].x == geometry->vertices[i].x && again->vertices[i].y == geometry->vertices[i].y &&
              again->vertices[i].z == geometry->vertices[i].z);
    }
    CHECK(again->faces == geometry->faces);

    // The mirrored copy's normal (1,0,0) goes through the inverse transpose: (-1,0,0).
    {
        const auto & primitive = root["meshes"][root["nodes"][5]["mesh"].get<std::size_t>()]["primitives"][0];
        const auto & accessor = root["accessors"][primitive["attributes"]["NORMAL"].get<std::size_t>()];
        const auto & view = root["bufferViews"][accessor["bufferView"].get<std::size_t>()];
        std::uint32_t json_size = 0;
        std::memcpy(&json_size, out->data() + 12, 4);
        const std::byte * bin = out->data() + 20 + json_size + 8;
        float n[3];
        std::memcpy(n, bin + view["byteOffset"].get<std::size_t>(), sizeof(n));
        CHECK(std::abs(n[0] + 1.0F) < 1e-6F && std::abs(n[1]) < 1e-6F && std::abs(n[2]) < 1e-6F);
    }

    // Control: the old writer rebuilds the mesh and keeps one material and no UVs.
    auto old = skintokens::save_skinned_animation_glb_memory(*geometry, binding, rest);
    CHECK(old);
    const json old_root = read_json(*old);
    CHECK(old_root["materials"].size() == 1U && !old_root.contains("textures"));
    CHECK(old_root["animations"][0]["name"] == "RESET");

    // A rigged file with one primitive per material reads back as one binding, and a clip added
    // to it keeps the materials and the RESET clip.
    {
        const auto path = std::filesystem::temp_directory_path() / "skintokens-source-glb-test.glb";
        std::FILE * file = std::fopen(path.string().c_str(), "wb");
        CHECK(file != nullptr);
        std::fwrite(out->data(), 1, out->size(), file);
        std::fclose(file);
        auto asset = skintokens::load_skinned_glb_file(path);
        std::filesystem::remove(path);
        if (!asset) { std::fprintf(stderr, "reload: %s\n", asset.error().message.c_str()); return 1; }
        CHECK(asset->binding.joints.size() == geometry->vertices.size());
        CHECK(asset->binding.weights[0][0] > 0.74F && asset->binding.weights[0][0] < 0.76F);
        skintokens::motion walk = rest;
        walk.name = "walk";
        walk.frames = 3U;
        walk.root_translations.assign(3U, binding.rig.rest_positions.front());
        walk.local_rotations.assign(3U * binding.rig.names.size(), {});
        auto with_walk = skintokens::add_animation_glb_memory(*out, walk);
        CHECK(with_walk);
        const json walked = read_json(*with_walk);
        CHECK(walked["animations"].size() == 2U && walked["animations"][0]["name"] == "walk");
        CHECK(walked["animations"][1]["name"] == "RESET");
        CHECK(walked["materials"].size() == 2U && walked["images"].size() == 1U);
        walk.rig.names[1] = "NotInTheSkin";
        CHECK(!skintokens::add_animation_glb_memory(*out, walk));
    }

    // Joint frames: each joint's +Y runs along its bone, the deformation at rest is unchanged
    // (joint world matrix times inverse bind is the identity), and a clip reads back as written.
    {
        skintokens::skin body;
        body.rig.names = {"Hips", "Spine", "Head", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe", "LeftArm", "LeftHand"};
        body.rig.parents = {-1, 0, 1, 0, 3, 4, 5, 1, 7};
        body.rig.rest_positions = {{0, 1, 0}, {0, 1.2F, 0}, {0, 1.6F, 0}, {0.1F, 0.9F, 0}, {0.1F, 0.5F, 0},
                                   {0.1F, 0.1F, 0}, {0.1F, 0, 0.15F}, {0.2F, 1.5F, 0}, {0.6F, 1.5F, 0}};
        const auto frames = skintokens::joint_rest_frames(body.rig);
        CHECK(frames.size() == 9U);
        const auto turn = [](skintokens::quat q, skintokens::vec3 v) {
            const float tx = 2*(q.y*v.z - q.z*v.y), ty = 2*(q.z*v.x - q.x*v.z), tz = 2*(q.x*v.y - q.y*v.x);
            return skintokens::vec3{v.x + q.w*tx + (q.y*tz - q.z*ty), v.y + q.w*ty + (q.z*tx - q.x*tz),
                                    v.z + q.w*tz + (q.x*ty - q.y*tx)};
        };
        const auto along = [&](std::size_t joint, std::size_t child) {
            const auto & a = body.rig.rest_positions[joint];
            const auto & c = body.rig.rest_positions[child];
            const float dx = c.x - a.x, dy = c.y - a.y, dz = c.z - a.z, l = std::sqrt(dx*dx + dy*dy + dz*dz);
            const auto y = turn(frames[joint], {0, 1, 0});
            return (y.x*dx + y.y*dy + y.z*dz) / l > 0.9999F;
        };
        CHECK(along(0, 1) && along(1, 2) && along(3, 4) && along(4, 5) && along(5, 6) && along(7, 8));
        const auto leg_forward = turn(frames[3], {0, 0, 1});
        CHECK(leg_forward.z > 0.9999F);  // a leg turns down about the forward axis: forward stays +Z
        body.joints.assign(geometry->vertices.size(), {0, 0, 0, 0});
        body.weights.assign(geometry->vertices.size(), {1, 0, 0, 0});
        for (std::size_t v = 0; v < geometry->vertices.size(); ++v) body.joints[v][0] = static_cast<std::uint16_t>(v % 9U);
        skintokens::motion clip;
        clip.name = "wave";
        clip.frames = 2U;
        clip.rig = body.rig;
        clip.root_translations.assign(2U, body.rig.rest_positions[0]);
        clip.local_rotations.assign(2U * 9U, {});
        for (std::size_t j = 0; j < 9U; ++j) {
            const float a = 0.1F + 0.07F * static_cast<float>(j);
            clip.local_rotations[9U + j] = {std::sin(a) * 0.6F, std::sin(a) * 0.64F, std::sin(a) * 0.48F, std::cos(a)};
        }
        auto written = skintokens::save_rigged_source_glb_memory(source, body, clip);
        CHECK(written);
        const json framed = read_json(*written);
        CHECK(framed["nodes"][framed["skins"][0]["joints"][3].get<std::size_t>()].contains("rotation"));
        const auto path = std::filesystem::temp_directory_path() / "skintokens-joint-frames-test.glb";
        std::FILE * file = std::fopen(path.string().c_str(), "wb");
        CHECK(file != nullptr);
        std::fwrite(written->data(), 1, written->size(), file);
        std::fclose(file);
        auto back = skintokens::load_kimodo_glb_file(path);
        std::filesystem::remove(path);
        CHECK(back && back->frames == 2U && back->rig.names.size() == 9U);
        for (std::size_t j = 0; j < 9U; ++j) {
            const auto & p = back->rig.rest_positions[j];
            const auto & q = body.rig.rest_positions[j];
            CHECK(std::abs(p.x - q.x) < 1e-5F && std::abs(p.y - q.y) < 1e-5F && std::abs(p.z - q.z) < 1e-5F);
            for (std::size_t f = 0; f < 2U; ++f) {
                const auto & r = back->local_rotations[f * 9U + j];
                const auto & w = clip.local_rotations[f * 9U + j];
                const float d = std::abs(r.x*w.x + r.y*w.y + r.z*w.z + r.w*w.w);
                if (d < 0.99999F) { std::fprintf(stderr, "joint %zu frame %zu reads back %.6f\n", j, f, d); return 1; }
            }
        }
    }

    // A skinned mesh under a turned root (Sketchfab's Z-up to Y-up node) is placed by its joint,
    // not by that node: glTF ignores a skinned mesh's node transform. The rig and the rigged file
    // must keep the source's up and forward.
    {
        const float positions[] = {0,0,0, 1,0,0, 0,1,0};
        const std::uint8_t joints[] = {0,0,0,0, 0,0,0,0, 0,0,0,0};
        const float weights[] = {1,0,0,0, 1,0,0,0, 1,0,0,0};
        const std::uint16_t indices[] = {0,1,2};
        std::vector<std::byte> bin;
        for (float v : positions) append(bin, v);   // 0..36
        for (auto v : joints) append(bin, v);       // 36..48
        for (float v : weights) append(bin, v);     // 48..96
        for (auto v : indices) append(bin, v);      // 96..102
        const std::string text = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,2]}],
"nodes":[{"name":"Sketchfab_model","matrix":[1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1],"children":[1]},
 {"name":"Body","mesh":0,"skin":0},{"name":"Bone"}],
"skins":[{"joints":[2]}],
"meshes":[{"primitives":[{"attributes":{"POSITION":0,"JOINTS_0":1,"WEIGHTS_0":2},"indices":3}]}],
"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
 {"bufferView":1,"componentType":5121,"count":3,"type":"VEC4"},
 {"bufferView":2,"componentType":5126,"count":3,"type":"VEC4"},
 {"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}],
"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":12},
 {"buffer":0,"byteOffset":48,"byteLength":48},{"buffer":0,"byteOffset":96,"byteLength":6}],
"buffers":[{"byteLength":104}]})";
        const auto turned = pack(text, bin);
        auto placed = skintokens::load_glb_memory(turned);
        if (!placed) { std::fprintf(stderr, "turned root: %s\n", placed.error().message.c_str()); return 1; }
        CHECK(placed->vertices.size() == 3U);
        const auto same = [](skintokens::vec3 a, float x, float y, float z) {
            return std::abs(a.x - x) < 1e-6F && std::abs(a.y - y) < 1e-6F && std::abs(a.z - z) < 1e-6F;
        };
        // The vertex at +Y stays at +Y (the node matrix would move it to -Z).
        CHECK(same(placed->vertices[2], 0, 1, 0) && same(placed->vertices[1], 1, 0, 0));
        skintokens::skin one;
        one.rig.names = {"Hips"};
        one.rig.parents = {-1};
        one.rig.rest_positions = {{0, 0, 0}};
        one.joints.assign(3U, {0, 0, 0, 0});
        one.weights.assign(3U, {1, 0, 0, 0});
        skintokens::motion still;
        still.frames = 1U;
        still.rig = one.rig;
        still.root_translations.assign(1U, one.rig.rest_positions.front());
        still.local_rotations.assign(1U, {});
        auto rigged = skintokens::save_rigged_source_glb_memory(turned, one, still);
        if (!rigged) { std::fprintf(stderr, "turned writer: %s\n", rigged.error().message.c_str()); return 1; }
        auto back = skintokens::load_glb_memory(*rigged);
        CHECK(back && back->vertices.size() == 3U && same(back->vertices[2], 0, 1, 0));
    }

    // A binding for a different mesh is refused.
    auto short_binding = binding;
    short_binding.joints.pop_back();
    short_binding.weights.pop_back();
    CHECK(!skintokens::save_rigged_source_glb_memory(source, short_binding, rest));
    std::puts("source-glb-test: PASS");
    return 0;
}
