// match_vrm1_humanoid on an authored skeleton: a root above the hips, a skirt bone on the hips
// (four hip children, which the generated matcher refuses), no upperChest and no toes. Roles are
// VRM 1.0 Humanoid roles with glTF node indices (KHR_character_skeleton_mapping / VRMC_vrm);
// joints carry node indices 100 + joint. Controls: the generated matcher refuses the skeleton, a
// missing required bone fails, a role on a node that is not a joint fails, and joint names do not
// matter (all renamed: the same mapping).
#include <skintokens/skintokens.hpp>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct joint {
    const char *name;
    int parent;
    skintokens::vec3 at;
};

skintokens::skeleton authored() {
    // Parents precede children, as the skeleton loader writes them.
    const std::vector<joint> joints{
        {"Root", -1, {0, 0, 0}},
        {"pelvis", 0, {0, 1, 0}},
        {"skirt", 1, {0, .9F, .1F}},
        {"tummy", 1, {0, 1.1F, 0}},
        {"chest", 3, {0, 1.3F, 0}},
        {"neck", 4, {0, 1.5F, 0}},
        {"head", 5, {0, 1.6F, 0}},
        {"shoulder.l", 4, {.1F, 1.45F, 0}},
        {"arm.l", 7, {.2F, 1.45F, 0}},
        {"forearm.l", 8, {.45F, 1.45F, 0}},
        {"hand.l", 9, {.7F, 1.45F, 0}},
        {"shoulder.r", 4, {-.1F, 1.45F, 0}},
        {"arm.r", 11, {-.2F, 1.45F, 0}},
        {"forearm.r", 12, {-.45F, 1.45F, 0}},
        {"hand.r", 13, {-.7F, 1.45F, 0}},
        {"thigh.l", 1, {.1F, .95F, 0}},
        {"shin.l", 15, {.1F, .5F, 0}},
        {"foot.l", 16, {.1F, .08F, 0}},
        {"thigh.r", 1, {-.1F, .95F, 0}},
        {"shin.r", 18, {-.1F, .5F, 0}},
        {"foot.r", 19, {-.1F, .08F, 0}},
    };
    skintokens::skeleton rig;
    for (const auto &j : joints) {
        rig.names.emplace_back(j.name);
        rig.parents.push_back(j.parent);
        rig.rest_positions.push_back(j.at);
        rig.source_nodes.push_back(100U + rig.source_nodes.size());
    }
    return rig;
}

std::vector<std::pair<std::string, std::size_t>> bones() {
    // Role -> node index (joint index + 100), in the joint order of authored().
    return {{"hips", 101}, {"spine", 103}, {"chest", 104}, {"neck", 105}, {"head", 106},
        {"leftShoulder", 107}, {"leftUpperArm", 108}, {"leftLowerArm", 109}, {"leftHand", 110},
        {"rightShoulder", 111}, {"rightUpperArm", 112}, {"rightLowerArm", 113}, {"rightHand", 114},
        {"leftUpperLeg", 115}, {"leftLowerLeg", 116}, {"leftFoot", 117},
        {"rightUpperLeg", 118}, {"rightLowerLeg", 119}, {"rightFoot", 120}};
}
} // namespace

int main() {
    const auto rig = authored();
    check(!skintokens::match_generated_humanoid(rig).has_value(),
        "control: the generated matcher refuses a root above the hips");

    auto mapping = skintokens::match_vrm1_humanoid(rig, bones());
    check(mapping.has_value(), "the VRM bones map the authored skeleton");
    if (mapping) {
        const auto &m = mapping->soma_to_generated;
        check(m[0] == 1, "SOMA Hips -> pelvis (not the root above it)");
        check(m[1] == 3, "SOMA Spine1 -> VRM spine");
        check(m[2] == -1 && m[3] == 4, "no upperChest: Spine2 unmapped, SOMA Chest -> VRM chest");
        check(m[4] == 5 && m[5] == -1 && m[6] == 6, "neck and head; Neck2 unmapped");
        check(m[10] == 7 && m[11] == 8 && m[12] == 9 && m[13] == 10, "left shoulder, arm, forearm, hand");
        check(m[16] == 11 && m[19] == 14, "right shoulder and hand");
        check(m[22] == 15 && m[23] == 16 && m[24] == 17 && m[25] == -1, "left leg; no toes: ToeBase unmapped");
        check(m[26] == 18 && m[28] == 20 && m[29] == -1, "right leg; no toes");
        check(mapping->generated_roles[2] == skintokens::semantic_role::unmapped, "the skirt bone stays unmapped");
        check(mapping->report.mapped_core_joints == 19U, "19 joints mapped");
    }

    auto no_hand = bones();
    no_hand.erase(no_hand.begin() + 8);
    check(!skintokens::match_vrm1_humanoid(rig, no_hand).has_value(), "control: a missing leftHand fails");

    auto renamed = rig;
    for (auto &n : renamed.names) n = "x";
    auto by_index = skintokens::match_vrm1_humanoid(renamed, bones());
    check(by_index && mapping && by_index->soma_to_generated == mapping->soma_to_generated,
        "control: joint names are not used (all renamed, same mapping)");

    auto wrong = bones();
    wrong[0].second = 999U;
    check(!skintokens::match_vrm1_humanoid(rig, wrong).has_value(), "control: hips on a node that is not a joint fails");

    auto no_nodes = rig;
    no_nodes.source_nodes.clear();
    check(!skintokens::match_vrm1_humanoid(no_nodes, bones()).has_value(), "control: a skeleton without node indices fails");

    std::printf("%s\n", failures ? "VRM-HUMANOID-FAIL" : "VRM-HUMANOID-OK");
    return failures ? 1 : 0;
}
