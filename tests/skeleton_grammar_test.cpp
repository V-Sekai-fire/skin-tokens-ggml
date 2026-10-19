// The skeleton grammar stops a skeleton at the 256 joints the skin codes cover. Without the cap a
// sampled beam could grow past it and never finish; with every beam doing so, a rig failed only at
// the token limit (mint_-_neverness_to_everness.glb: 371 joints after 41 minutes).
#include "skeleton_grammar.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    using namespace skintokens::detail;
    const auto has = [](const std::vector<std::int32_t> &v, std::int32_t t) { return std::find(v.begin(), v.end(), t) != v.end(); };
    std::vector<std::int32_t> tokens{257, 266}; // BOS, class
    for (std::size_t joint = 0; joint + 1U < max_skeleton_joints; ++joint) tokens.insert(tokens.end(), {1, 2, 3});
    // Control: below the cap a skeleton may add a joint, a part, a branch, or end.
    auto parsed = parse_skeleton(tokens);
    CHECK(parsed.joints == max_skeleton_joints - 1U);
    auto allowed = allowed_skeleton_tokens(parsed);
    CHECK(has(allowed, 0) && has(allowed, 256) && has(allowed, 260) && has(allowed, 258));
    // At the cap the skeleton may only end.
    tokens.insert(tokens.end(), {1, 2, 3});
    parsed = parse_skeleton(tokens);
    CHECK(parsed.joints == max_skeleton_joints);
    allowed = allowed_skeleton_tokens(parsed);
    CHECK(allowed == std::vector<std::int32_t>{258});
    // A joint being written finishes: its second and third coordinates stay allowed.
    std::vector<std::int32_t> mid{257, 266};
    for (std::size_t joint = 0; joint + 1U < max_skeleton_joints; ++joint) mid.insert(mid.end(), {1, 2, 3});
    mid.push_back(1);
    CHECK(has(allowed_skeleton_tokens(parse_skeleton(mid)), 2));
    std::puts("skeleton-grammar-test: PASS");
    return 0;
}
