// Replays the point-cloud-to-vertex weight transfer from a transfer trace, so
// transfer changes can be measured without rerunning the model.
//
//   guest: opts "... trace=1" makes the rig program save <out>.stkt beside <out>
//   CLI:   SKINTOKENS_DUMP_TRANSFER=<file.stkt> skintokens-cli rig ...
//   skintokens-transfer-replay TRACE.stkt RIGGED.glb OUT.glb [--raw]
//          [--geodesic R] [--facing DEG] [--outlier L1] [--support S] [--reach F]
//
// RIGGED.glb supplies the mesh and the skeleton (any rig of the same run).
// --raw writes the upstream eight-neighbour interpolation instead; it must
// be identical to a robust=0 rig of the same run (the control).
#include <skintokens/skintokens.hpp>

#include "binding.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: skintokens-transfer-replay TRACE.stkt RIGGED.glb OUT.glb [--raw]"
            " [--geodesic R] [--facing DEG] [--outlier L1] [--support S] [--reach F]\n");
        return 2;
    }
    skintokens::detail::robust_transfer_options options;
    bool raw_only = false;
    for (int i = 4; i < argc; ++i) {
        const std::string_view key{argv[i]};
        if (key == "--raw") {
            raw_only = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            return 2;
        }
        const float value = std::strtof(argv[++i], nullptr);
        if (key == "--geodesic") options.geodesic_ratio = value;
        else if (key == "--facing") options.max_normal_degrees = value;
        else if (key == "--outlier") options.outlier_l1 = value;
        else if (key == "--support") options.support_share = value;
        else if (key == "--reach") options.distance_fraction = value;
        else if (key == "--smooth") options.smooth_iterations = static_cast<std::size_t>(value);
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i - 1]);
            return 2;
        }
    }
    std::ifstream stream(argv[1], std::ios::binary);
    const std::vector<char> chars{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    skintokens::detail::transfer_inputs_storage in;
    if (!skintokens::detail::deserialize_transfer_inputs(
            std::span{reinterpret_cast<const std::byte *>(chars.data()), chars.size()}, in)) {
        std::fprintf(stderr, "FAIL %s is not a transfer trace\n", argv[1]);
        return 1;
    }
    auto asset = skintokens::load_skinned_glb_file(argv[2]);
    if (!asset) {
        std::fprintf(stderr, "FAIL load: %s\n", asset.error().message.c_str());
        return 1;
    }
    if (asset->geometry.vertices.size() != in.normalized_vertices.size() ||
        asset->binding.rig.names.size() != in.dense_joint_weights.size()) {
        std::fprintf(stderr, "FAIL trace and GLB disagree: vertices %zu/%zu joints %zu/%zu\n",
            in.normalized_vertices.size(), asset->geometry.vertices.size(),
            in.dense_joint_weights.size(), asset->binding.rig.names.size());
        return 1;
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = raw_only ?
        skintokens::detail::integrate_learned_binding(asset->binding.rig, in.normalized_vertices,
            in.sampled_points, in.dense_joint_weights) :
        skintokens::detail::integrate_robust_binding(asset->binding.rig, in.normalized_vertices, in.vertex_normals,
            in.faces, in.sampled_points, in.sampled_normals, in.sampled_faces, in.dense_joint_weights, options);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto &r = result.transfer;
    std::printf(
        "REPLAY %s welded=%zu matched=%zu outliers=%zu inpainted=%zu fallback=%zu trimmed_parts=%zu"
        " solved_joints=%zu iterations=%zu residual=%.1e ms=%.0f\n",
        raw_only ? "raw" : "robust", r.welded, r.matched, r.outliers, r.inpainted, r.fallback, r.trimmed_parts,
        r.solved_joints, r.iterations, static_cast<double>(r.residual), ms);
    asset->binding.joints = result.joints;
    asset->binding.weights = result.weights;
    skintokens::motion rest;
    rest.frames = 1U;
    rest.rig = asset->binding.rig;
    rest.root_translations.assign(1U, rest.rig.rest_positions.front());
    rest.local_rotations.assign(rest.rig.names.size(), {});
    auto saved = skintokens::save_skinned_animation_glb_file(argv[3], asset->geometry, asset->binding, rest);
    if (!saved) {
        std::fprintf(stderr, "FAIL save: %s\n", saved.error().message.c_str());
        return 1;
    }
    std::printf("REPLAY-OK %s\n", argv[3]);
    return 0;
}
