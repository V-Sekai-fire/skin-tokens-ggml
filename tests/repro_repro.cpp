// Repro for the authored-rig "add motion" path (459 joints, 3 skins).
// Builds a valid SOMA30 rest clip and calls add_soma30_motion_glb_memory on a real rigged GLB.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <skintokens/skintokens.hpp>

using namespace skintokens;

std::vector<std::byte> read_all(const std::filesystem::path &p) {
    std::vector<std::byte> b;
    std::ifstream in(p, std::ios::binary);
    in.seekg(0, std::ios::end);
    b.resize(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(b.data()), b.size());
    return b;
}

// Minimal valid SOMA30 rest clip: exactly 30 joints, parents precede children, one root.
static motion soma30_rest() {
    motion m;
    m.rig.names = {
        "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"};
    std::vector<int32_t> p(30);
    p[0] = -1;
    p[1] = 0;
    p[2] = 1;
    p[3] = 2;
    p[4] = 3;
    p[5] = 4;
    p[6] = 5;
    p[7] = 6;
    p[8] = 6;
    p[9] = 6;
    p[10] = 3;
    p[11] = 10;
    p[12] = 11;
    p[13] = 12;
    p[14] = 13;
    p[15] = 13;
    p[16] = 3;
    p[17] = 16;
    p[18] = 17;
    p[19] = 18;
    p[20] = 19;
    p[21] = 19;
    p[22] = 0;
    p[23] = 22;
    p[24] = 23;
    p[25] = 24;
    p[26] = 0;
    p[27] = 26;
    p[28] = 27;
    p[29] = 28;
    m.rig.parents = std::move(p);
    static const float x[30] = {0, 0, 0, 0, 0, 0, 0, 0, .05f, -.05f, .25f, .5f, .75f, .9f, .95f, 1.0f,
        -.25f, -.5f, -.75f, -.9f, -.95f, -1.0f, .1f, .1f, .1f, .1f, -.1f, -.1f, -.1f, -.1f};
    static const float y[30] = {0, 1.0f, 1.6f, 2.0f, 2.3f, 2.45f, 2.6f, 2.65f, 2.68f, 2.68f,
        2.1f, 2.0f, 1.8f, 1.7f, 1.7f, 1.7f, 2.1f, 2.0f, 1.8f, 1.7f, 1.7f, 1.7f,
        1.0f, .8f, .6f, .5f, 1.0f, .8f, .6f, .5f};
    for (int i = 0; i < 30; i++) m.rig.rest_positions.push_back(vec3{x[i], y[i], 0.0F});
    m.frames = 4U;
    m.frames_per_second = 30.0F;
    m.root_translations.assign(4, vec3{0, 0, 0});
    m.local_rotations.assign(4 * 30, quat{}); // identity -> rest pose
    return m;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::printf("usage: %s <rigged.glb>\n", argv[0]);
        return 2;
    }
    std::vector<std::byte> bytes = read_all(std::filesystem::path(argv[1]));
    if (bytes.empty()) {
        std::printf("empty GLB\n");
        return 2;
    }

    auto skeleton = load_skeleton_glb_memory(bytes);
    if (!skeleton) {
        std::printf("FAIL skeleton load: %s\n", skeleton.error().message.c_str());
        return 3;
    }
    std::printf("skeleton: %zu joints\n", skeleton->rig.names.size());

    auto roles = vrm1_humanoid_roles_glb_memory(bytes);
    if (!roles) {
        std::printf("FAIL roles: %s\n", roles.error().message.c_str());
        return 4;
    }
    std::printf("roles: %zu bones\n", roles->size());

    humanoid_match_options match;
    retarget_options transfer;
    match.minimum_confidence = 0.5F;
    transfer.minimum_confidence = 0.5F;
    auto added = add_soma30_motion_glb_memory(bytes, soma30_rest(), match, transfer);
    if (!added) {
        std::printf("FAIL add motion: %s\n", added.error().message.c_str());
        return 5;
    }
    std::printf("SUCCESS  add motion -> %zu bytes\n", added->first.size());
    std::printf("  confidence %.3f, core score %.3f, mapped core joints %zu\n",
        added->second.confidence, added->second.core_score, added->second.mapped_core_joints);
    return 0;
}
