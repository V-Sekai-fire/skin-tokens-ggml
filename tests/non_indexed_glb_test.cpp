// A primitive without indices is a plain triangle list (glTF 2.0 3.7.2.1): vertices 3k, 3k+1 and
// 3k+2 form triangle k. The loader, the skin reader and the source writer must all read it.
#include <skintokens/skintokens.hpp>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;

template<class T>
void append(std::vector<std::byte> &output, const T &value) {
    const std::size_t offset = output.size();
    output.resize(offset + sizeof(T));
    std::memcpy(output.data() + offset, &value, sizeof(T));
}

std::vector<std::byte> pack(std::string text, std::vector<std::byte> binary) {
    while (text.size() % 4U) text.push_back(' ');
    while (binary.size() % 4U) binary.push_back(std::byte{});
    std::vector<std::byte> out;
    append(out, 0x46546C67U);
    append(out, 2U);
    append(out, static_cast<std::uint32_t>(12U + 8U + text.size() + 8U + binary.size()));
    append(out, static_cast<std::uint32_t>(text.size()));
    append(out, 0x4E4F534AU);
    for (char c : text) out.push_back(static_cast<std::byte>(c));
    append(out, static_cast<std::uint32_t>(binary.size()));
    append(out, 0x004E4942U);
    out.insert(out.end(), binary.begin(), binary.end());
    return out;
}

// Two triangles (six vertices) placed by a plain node and a mirrored one. With `indexed` the same
// triangles come through an index accessor {0..5}; `count` sets the POSITION count.
std::vector<std::byte> fixture(bool indexed, std::size_t count) {
    const float positions[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1};
    const std::uint16_t indices[] = {0, 1, 2, 3, 4, 5};
    std::vector<std::byte> bin;
    for (float v : positions) append(bin, v); // 0..72
    for (std::uint16_t v : indices) append(bin, v); // 72..84
    std::string primitive = R"({"attributes":{"POSITION":0})";
    if (indexed) primitive += R"(,"indices":1)";
    primitive += "}";
    const std::string text = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],
"nodes":[{"name":"Plain","mesh":0},{"name":"Mirror","mesh":0,"scale":[-1,1,1]}],
"meshes":[{"primitives":[)" +
        primitive + R"(]}],
"accessors":[{"bufferView":0,"componentType":5126,"count":)" +
        std::to_string(count) + R"(,"type":"VEC3"},
 {"bufferView":1,"componentType":5123,"count":6,"type":"SCALAR"}],
"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":72},{"buffer":0,"byteOffset":72,"byteLength":12}],
"buffers":[{"byteLength":84}]})";
    return pack(text, bin);
}

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

const std::vector<skintokens::triangle> expected_faces = {{0, 1, 2}, {3, 4, 5}, {6, 8, 7}, {9, 11, 10}};
} // namespace

int main() {
    // Control: the indexed form of the same mesh.
    const std::vector<std::byte> indexed = fixture(true, 6U);
    skintokens::result<skintokens::mesh> control = skintokens::load_glb_memory(indexed);
    CHECK(control && control->vertices.size() == 12U && control->faces == expected_faces);

    const std::vector<std::byte> source = fixture(false, 6U);
    skintokens::result<skintokens::mesh> geometry = skintokens::load_glb_memory(source);
    if (!geometry) {
        std::fprintf(stderr, "non-indexed load: %s\n", geometry.error().message.c_str());
        return 1;
    }
    CHECK(geometry->vertices.size() == 12U && geometry->faces.size() == 4U);
    CHECK(geometry->faces == expected_faces); // the mirrored copy keeps its winding
    for (std::size_t i = 0; i < 12U; ++i) {
        CHECK(geometry->vertices[i].x == control->vertices[i].x && geometry->vertices[i].y == control->vertices[i].y &&
            geometry->vertices[i].z == control->vertices[i].z);
    }

    // A non-indexed POSITION count that is not a multiple of three is refused.
    CHECK(!skintokens::load_glb_memory(fixture(false, 5U)));

    // The writer skins the same faces, and the rigged file reads back as one binding.
    skintokens::skin binding;
    binding.rig.names = {"Hips"};
    binding.rig.parents = {-1};
    binding.rig.rest_positions = {{0, 0, 0}};
    binding.joints.assign(12U, {0, 0, 0, 0});
    binding.weights.assign(12U, {1, 0, 0, 0});
    skintokens::motion rest;
    rest.frames = 1U;
    rest.rig = binding.rig;
    rest.root_translations.assign(1U, binding.rig.rest_positions.front());
    rest.local_rotations.assign(1U, {});
    skintokens::result<std::vector<std::byte>> rigged = skintokens::save_rigged_source_glb_memory(source, binding, rest);
    if (!rigged) {
        std::fprintf(stderr, "non-indexed writer: %s\n", rigged.error().message.c_str());
        return 1;
    }
    skintokens::result<skintokens::mesh> back = skintokens::load_glb_memory(*rigged);
    CHECK(back && back->vertices.size() == 12U && back->faces == expected_faces);

    const std::filesystem::path path = std::filesystem::temp_directory_path() / "skintokens-non-indexed-test.glb";
    std::FILE *file = std::fopen(path.string().c_str(), "wb");
    CHECK(file != nullptr);
    std::fwrite(rigged->data(), 1, rigged->size(), file);
    std::fclose(file);
    skintokens::result<skintokens::skinned_asset> asset = skintokens::load_skinned_glb_file(path);
    std::filesystem::remove(path);
    if (!asset) {
        std::fprintf(stderr, "non-indexed reload: %s\n", asset.error().message.c_str());
        return 1;
    }
    CHECK(asset->geometry.faces == expected_faces && asset->binding.weights.size() == 12U);
    std::puts("non-indexed-glb-test: PASS");
    return 0;
}
