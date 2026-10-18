// save_rigged_source_glb_memory keeps the source GLB's materials, textures and UVs, and the old
// writer (the control) does not.
#include <skintokens/skintokens.hpp>
#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
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

    // A binding for a different mesh is refused.
    auto short_binding = binding;
    short_binding.joints.pop_back();
    short_binding.weights.pop_back();
    CHECK(!skintokens::save_rigged_source_glb_memory(source, short_binding, rest));
    std::puts("source-glb-test: PASS");
    return 0;
}
