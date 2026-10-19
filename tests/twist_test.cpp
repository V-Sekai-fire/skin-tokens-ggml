// add_twist_joints: four helpers per long limb segment, and the segment's weight split among them
// by where each vertex lies along the segment (a ggml graph, here on the CPU device).
#include <skintokens/skintokens.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

int main() {
    skintokens::skeleton soma30;
    soma30.names = {"Hips","Spine1","Spine2","Chest","Neck1","Neck2","Head","Jaw","LeftEye","RightEye",
        "LeftShoulder","LeftArm","LeftForeArm","LeftHand","LeftHandThumbEnd","LeftHandMiddleEnd",
        "RightShoulder","RightArm","RightForeArm","RightHand","RightHandThumbEnd","RightHandMiddleEnd",
        "LeftLeg","LeftShin","LeftFoot","LeftToeBase","RightLeg","RightShin","RightFoot","RightToeBase"};
    soma30.parents = {-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28};
    soma30.rest_positions = {{0,1,0},{0,1.15F,0},{0,1.3F,0},{0,1.45F,0},{0,1.58F,0},{0,1.66F,0},{0,1.78F,0},
        {0,1.72F,.04F},{-.03F,1.8F,.08F},{.03F,1.8F,.08F},{-.12F,1.46F,0},{-.35F,1.44F,0},{-.58F,1.35F,0},
        {-.74F,1.25F,0},{-.84F,1.2F,.05F},{-.9F,1.2F,0},{.12F,1.46F,0},{.35F,1.44F,0},{.58F,1.35F,0},
        {.74F,1.25F,0},{.84F,1.2F,.05F},{.9F,1.2F,0},{-.12F,.92F,0},{-.13F,.5F,0},{-.13F,.12F,.02F},
        {-.13F,.03F,.18F},{.12F,.92F,0},{.13F,.5F,0},{.13F,.12F,.02F},{.13F,.03F,.18F}};
    auto rig = skintokens::make_mixamo52_rig(soma30);
    CHECK(rig && rig->names.size() == 52U);
    const auto index = [&](const std::string & name) {
        for (std::size_t i = 0; i < rig->names.size(); ++i) if (rig->names[i] == name) return static_cast<int>(i);
        return -1;
    };
    const int forearm = index("mixamorig:LeftForeArm"), hand = index("mixamorig:LeftHand"), hips = index("mixamorig:Hips");
    CHECK(forearm >= 0 && hand >= 0 && hips >= 0);
    const auto a = rig->rest_positions[static_cast<std::size_t>(forearm)];
    const auto b = rig->rest_positions[static_cast<std::size_t>(hand)];

    // Vertices along the left forearm (all weight on the forearm joint), and one on the hips.
    const float along[] = {-0.1F, 0.0F, 0.05F, 0.2F, 0.5F, 0.8F, 0.95F, 1.0F, 1.1F};
    skintokens::mesh geometry;
    skintokens::skin binding;
    binding.rig = *rig;
    for (float t : along) {
        geometry.vertices.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)});
        binding.joints.push_back({static_cast<std::uint16_t>(forearm), 0, 0, 0});
        binding.weights.push_back({1.0F, 0.0F, 0.0F, 0.0F});
    }
    geometry.vertices.push_back({0.0F, 1.0F, 0.0F});
    binding.joints.push_back({static_cast<std::uint16_t>(hips), static_cast<std::uint16_t>(forearm), 0, 0});
    binding.weights.push_back({0.75F, 0.25F, 0.0F, 0.0F});
    // The vertex at the hips lies past the forearm's start: its quarter goes to Twist1.

    skintokens::runtime_options cpu;
    cpu.device = skintokens::device_kind::cpu;
    auto twisted = skintokens::add_twist_joints(geometry, binding, cpu);
    if (!twisted) { std::fprintf(stderr, "twist: %s\n", twisted.error().message.c_str()); return 1; }
    const auto & out = *twisted;
    CHECK(out.segments.size() == 8U);
    CHECK(out.binding.rig.names.size() == 52U + 32U);
    const skintokens::twist_segment * fore = nullptr;
    for (const auto & s : out.segments) if (s.start == forearm) fore = &s;
    CHECK(fore != nullptr && fore->end == hand && !fore->reverse);
    const auto & f = skintokens::twist_fractions();
    for (std::size_t k = 0; k < 4U; ++k) {
        const auto helper = static_cast<std::size_t>(fore->helpers[k]);
        CHECK(out.binding.rig.names[helper] == "mixamorig:LeftForeArmTwist" + std::to_string(k + 1U));
        CHECK(out.binding.rig.parents[helper] == forearm);
        const auto p = out.binding.rig.rest_positions[helper];
        CHECK(std::abs(p.x - (a.x + f[k] * (b.x - a.x))) < 1e-6F && std::abs(p.y - (a.y + f[k] * (b.y - a.y))) < 1e-6F);
    }
    // Every vertex's shares on every segment sum to 1, and match the tent functions computed here.
    const std::size_t rows = out.segments.size() * 4U;
    const auto tent = [&](float t, std::size_t k) {
        const float lo = k == 0U ? -1e9F : f[k - 1U], hi = k == 3U ? 1e9F : f[k + 1U];
        const float rise = k == 0U ? 1e9F : (t - lo) / (f[k] - lo), fall = k == 3U ? 1e9F : (hi - t) / (hi - f[k]);
        return std::clamp(std::min(rise, fall), 0.0F, 1.0F);
    };
    std::size_t fore_row = 0;
    for (std::size_t s = 0; s < out.segments.size(); ++s) if (out.segments[s].start == forearm) fore_row = s * 4U;
    for (std::size_t v = 0; v < geometry.vertices.size(); ++v) {
        for (std::size_t s = 0; s < out.segments.size(); ++s) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < 4U; ++k) sum += out.shares[v * rows + s * 4U + k];
            CHECK(std::abs(sum - 1.0F) < 1e-5F);
        }
        const auto & p = geometry.vertices[v];
        const float d2 = (b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y) + (b.z-a.z)*(b.z-a.z);
        const float t = ((p.x-a.x)*(b.x-a.x) + (p.y-a.y)*(b.y-a.y) + (p.z-a.z)*(b.z-a.z)) / d2;
        for (std::size_t k = 0; k < 4U; ++k) CHECK(std::abs(out.shares[v * rows + fore_row + k] - tent(t, k)) < 1e-5F);
    }
    // Named cases: ends go wholly to the end helpers; halfway splits Twist2 and Twist3 evenly.
    const auto weight_on = [&](std::size_t v, int joint) {
        float w = 0.0F;
        for (std::size_t slot = 0; slot < 4U; ++slot) if (out.binding.joints[v][slot] == joint) w += out.binding.weights[v][slot];
        return w;
    };
    CHECK(std::abs(weight_on(0, fore->helpers[0]) - 1.0F) < 1e-5F);  // t = -0.1
    CHECK(std::abs(weight_on(8, fore->helpers[3]) - 1.0F) < 1e-5F);  // t = 1.1
    CHECK(std::abs(weight_on(4, fore->helpers[1]) - 0.5F) < 1e-4F && std::abs(weight_on(4, fore->helpers[2]) - 0.5F) < 1e-4F);
    CHECK(weight_on(4, forearm) == 0.0F);                             // the start joint gave its weight away
    CHECK(std::abs(weight_on(9, hips) - 0.75F) < 1e-6F);              // other joints keep theirs
    for (std::size_t v = 0; v < geometry.vertices.size(); ++v) {
        float sum = 0.0F;
        for (float w : out.binding.weights[v]) sum += w;
        CHECK(std::abs(sum - 1.0F) < 1e-5F);
    }

    // Control: a rig that is not a humanoid gets no helpers.
    skintokens::skin chain;
    chain.rig.names = {"a", "b", "c"};
    chain.rig.parents = {-1, 0, 1};
    chain.rig.rest_positions = {{0, 0, 0}, {0, 1, 0}, {0, 2, 0}};
    chain.joints = {{0, 0, 0, 0}};
    chain.weights = {{1, 0, 0, 0}};
    skintokens::mesh one;
    one.vertices = {{0, 0.5F, 0}};
    CHECK(!skintokens::add_twist_joints(one, chain, cpu));
    std::puts("twist-test: PASS");
    return 0;
}
