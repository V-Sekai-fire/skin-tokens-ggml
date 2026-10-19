#include <sstream>
#include <skintokens/skintokens.hpp>

#include <cstring>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <string_view>

namespace {

void usage() {
    std::cerr << "usage:\n"
              << "  skintokens-cli inspect MODEL_DIR [--device auto|cpu|vulkan]\n"
              << "  skintokens-cli rig MODEL_DIR MESH.{glb,t2mesh} OUTPUT.glb"
                 " [--device auto|cpu|vulkan] [--postprocess] [--no-robust-transfer]"
                 " [--beams N] [--temperature F] [--max-tokens N]\n"
              << "  skintokens-cli skin MODEL_DIR MESH.{glb,t2mesh} SKELETON.glb OUTPUT.glb"
                 " [--device auto|cpu|vulkan] [--fit none|global|articulated]"
                 " [--retarget-soma-to-mixamo52]"
                 " [--postprocess] [--beams N] [--temperature F] [--max-tokens N]\n"
              << "  skintokens-cli glb-info FILE.glb\n"
        "  skintokens-cli reframe IN.{glb,vrm} OUT.{glb,vrm}\n"
        "  skintokens-cli t-pose IN.vrm OUT.vrm\n"
              << "  skintokens-cli retarget-check SOMA30.glb [MIXAMO52.glb]\n"
              << "  skintokens-cli retarget-generated GENERATED.glb SOMA30.glb OUTPUT.glb"
                 " [--minimum-confidence F] [--map-fingers]\n"
              << "  skintokens-cli retarget-generated-check SOMA30.glb GENERATED.glb\n"
              << "  skintokens-cli add-motion RIGGED.glb CLIP_DIR loop|once SECONDS NAME OUTPUT.glb\n"
              << "  skintokens-cli humanoid GENERATED.glb\n"
              << "  skintokens-cli prepare-mixamo MESH.glb SOMA30.glb OUTPUT.glb\n"
              << "  skintokens-cli bind MODEL_DIR MESH.{glb,t2mesh} MOTION.glb OUTPUT.glb"
                 " [--device auto|cpu|vulkan] [--geometric|--supplied-skeleton]"
                 " [--fit none|global|articulated]"
                 " [--target-rig soma30|mixamo52]"
                 " [--postprocess]"
                 " [--beams N] [--temperature F] [--max-tokens N]\n";
}

skintokens::device_kind parse_device(std::string_view value) {
    if (value == "cpu") return skintokens::device_kind::cpu;
    if (value == "vulkan") return skintokens::device_kind::vulkan;
    return skintokens::device_kind::automatic;
}

skintokens::skeleton_fit parse_fit(std::string_view value) {
    if (value == "none") return skintokens::skeleton_fit::none;
    if (value == "articulated") return skintokens::skeleton_fit::articulated;
    if (value == "global") return skintokens::skeleton_fit::global_similarity;
    throw std::invalid_argument("fit must be none, global, or articulated");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    if (std::string_view{argv[1]} == "retarget-check") {
        auto animation = skintokens::load_kimodo_glb_file(argv[2]);
        if (!animation) { std::cerr << animation.error().message << '\n'; return 1; }
        auto mixamo = skintokens::make_mixamo52_rig(animation->rig);
        if (!mixamo) { std::cerr << mixamo.error().message << '\n'; return 1; }
        auto report = skintokens::validate_soma30_to_mixamo52(*animation, *mixamo);
        if (!report) { std::cerr << report.error().message << '\n'; return 1; }
        std::cout << std::fixed << std::setprecision(7)
                  << "isolated cases: " << report->isolated_cases << '\n'
                  << "isolated mean normalized position error: " << report->isolated_mean_position_error << '\n'
                  << "isolated max normalized position error: " << report->isolated_max_position_error << '\n'
                  << "clip mean normalized position error: " << report->motion_mean_position_error << '\n'
                  << "clip max normalized position error: " << report->motion_max_position_error << '\n'
                  << "foot velocity relative error: " << report->foot_velocity_relative_error << '\n';
        if (argc >= 4) {
            auto exported = skintokens::load_kimodo_glb_file(argv[3]);
            if (!exported) { std::cerr << exported.error().message << '\n'; return 1; }
            auto roundtrip = skintokens::compare_soma30_to_mixamo52(*animation, *exported);
            if (!roundtrip) { std::cerr << roundtrip.error().message << '\n'; return 1; }
            std::cout << "exported clip mean normalized position error: "
                      << roundtrip->motion_mean_position_error << '\n'
                      << "exported clip max normalized position error: "
                      << roundtrip->motion_max_position_error << '\n'
                      << "exported foot velocity relative error: "
                      << roundtrip->foot_velocity_relative_error << '\n';
        }
        return 0;
    }
    if (std::string_view{argv[1]} == "prepare-mixamo") {
        if (argc != 5) { usage(); return 2; }
        auto geometry = skintokens::load_glb_file(argv[2]);
        auto source = skintokens::load_kimodo_glb_file(argv[3]);
        if (!geometry) { std::cerr << geometry.error().message << '\n'; return 1; }
        if (!source) { std::cerr << source.error().message << '\n'; return 1; }
        auto fitted = skintokens::fit_motion_to_mesh(*geometry, *source);
        if (!fitted) { std::cerr << fitted.error().message << '\n'; return 1; }
        auto mixamo = skintokens::make_mixamo52_rig(fitted->rig);
        if (!mixamo) { std::cerr << mixamo.error().message << '\n'; return 1; }
        auto animation = skintokens::retarget_soma30_to_mixamo52(*fitted, *mixamo);
        if (!animation) { std::cerr << animation.error().message << '\n'; return 1; }
        if (geometry->vertices.size() < mixamo->names.size()) {
            std::cerr << "mesh has too few vertices for the covered-skeleton fixture\n";
            return 1;
        }
        skintokens::skin covered;
        covered.rig = *mixamo;
        covered.joints.assign(geometry->vertices.size(), {0U, 0U, 0U, 0U});
        covered.weights.assign(geometry->vertices.size(), {1.0F, 0.0F, 0.0F, 0.0F});
        // Upstream's predict transform trims unweighted leaves before
        // tokenization. Give every supplied joint a harmless 0.1% marker so
        // the comparison exercises the requested 52-joint topology.
        for (std::size_t joint = 1U; joint < mixamo->names.size(); ++joint) {
            covered.joints[joint] = {0U, static_cast<std::uint16_t>(joint), 0U, 0U};
            covered.weights[joint] = {0.999F, 0.001F, 0.0F, 0.0F};
        }
        auto saved = skintokens::save_skinned_animation_glb_file(
            argv[4], *geometry, covered, *animation);
        if (!saved) { std::cerr << saved.error().message << '\n'; return 1; }
        std::cout << "covered Mixamo52 comparison fixture written to " << argv[4] << '\n';
        return 0;
    }
    if (std::string_view{argv[1]} == "t-pose") {
        if (argc != 4) { usage(); return 2; }
        std::ifstream in(argv[2], std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<std::byte> bytes(raw.size());
        std::memcpy(bytes.data(), raw.data(), raw.size());
        auto out = skintokens::pose_to_t_pose_glb_memory(bytes);
        if (!out) { std::cerr << out.error().message << '\n'; return 1; }
        std::ofstream file(argv[3], std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char *>(out->data()), static_cast<std::streamsize>(out->size()));
        std::cout << "T-posed " << argv[2] << " -> " << argv[3] << ", " << out->size() << " bytes\n";
        return file ? 0 : 1;
    }
    if (std::string_view{argv[1]} == "reframe") {
        if (argc != 4) { usage(); return 2; }
        std::ifstream in(argv[2], std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<std::byte> bytes(raw.size());
        std::memcpy(bytes.data(), raw.data(), raw.size());
        auto out = skintokens::reframe_skin_joints_glb_memory(bytes);
        if (!out) { std::cerr << out.error().message << '\n'; return 1; }
        std::ofstream file(argv[3], std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char *>(out->data()), static_cast<std::streamsize>(out->size()));
        std::cout << "re-framed " << argv[2] << " -> " << argv[3] << ", " << out->size() << " bytes\n";
        return file ? 0 : 1;
    }
    if (std::string_view{argv[1]} == "glb-info") {
        if (argc != 3) { usage(); return 2; }
        auto info = skintokens::inspect_glb_file(argv[2]);
        if (!info) { std::cerr << info.error().message << '\n'; return 1; }
        const char * rig = info->rig == skintokens::rig_kind::soma30 ? "soma30" :
            info->rig == skintokens::rig_kind::mixamo52 ? "mixamo52" : "unknown";
        std::cout << "{\"hasMesh\":" << (info->has_mesh ? "true" : "false")
                  << ",\"hasSkin\":" << (info->has_skin ? "true" : "false")
                  << ",\"hasSkeleton\":" << (info->has_skeleton ? "true" : "false")
                  << ",\"hasAnimation\":" << (info->has_animation ? "true" : "false")
                  << ",\"jointCount\":" << info->joint_count
                  << ",\"frameCount\":" << info->frame_count
                  << ",\"framesPerSecond\":" << info->frames_per_second
                  << ",\"rigKind\":\"" << rig << "\"}\n";
        return 0;
    }
    if (std::string_view{argv[1]} == "humanoid") {
        if (argc != 3) { usage(); return 2; }
        auto asset = skintokens::load_skinned_glb_file(argv[2]);
        if (!asset) { std::cerr << asset.error().message << '\n'; return 1; }
        const std::string json = skintokens::humanoid_roles_json(asset->binding.rig);
        std::cout << json << '\n';
        return json.starts_with("{\"gates\":\"pass\"") ? 0 : 3;
    }
    if (std::string_view{argv[1]} == "retarget-generated") {
        if (argc < 5) { usage(); return 2; }
        skintokens::humanoid_match_options match_options;
        skintokens::retarget_options transfer_options;
        for (int i = 5; i < argc; ++i) {
            const std::string_view argument{argv[i]};
            if (argument == "--minimum-confidence" && i + 1 < argc) {
                match_options.minimum_confidence = std::stof(argv[++i]);
                transfer_options.minimum_confidence = match_options.minimum_confidence;
            } else if (argument == "--map-fingers") {
                transfer_options.fingers = skintokens::finger_transfer::map_soma_endpoints;
            } else {
                std::cerr << "unknown retarget-generated option: " << argument << '\n';
                return 2;
            }
        }
        auto report = skintokens::retarget_soma30_glb_file(
            argv[2], argv[3], argv[4], match_options, transfer_options);
        if (!report) { std::cerr << report.error().message << '\n'; return 1; }
        std::cout << std::fixed << std::setprecision(3)
                  << "retargeted SOMA30 motion onto generated rig: confidence "
                  << report->confidence << ", " << report->mapped_core_joints
                  << " core joints, " << report->ignored_terminal_joints
                  << " flexible hand joints\n";
        return 0;
    }
    if (std::string_view{argv[1]} == "add-motion") {
        // add-motion RIGGED.glb CLIP_DIR loop|once SECONDS NAME OUTPUT.glb [--cut-only OUT.f32]
        // CLIP_DIR holds motion.elf's transforms.f32 + skeleton.txt + parents.txt.
        if (argc < 8) { usage(); return 2; }
        const std::filesystem::path dir{argv[3]};
        std::ifstream sk(dir / "skeleton.txt");
        std::vector<std::string> lines;
        for (std::string l; std::getline(sk, l);) lines.push_back(l);
        if (lines.size() < 4) { std::cerr << "skeleton.txt is short\n"; return 1; }
        std::istringstream head(lines[2]);
        std::string word;
        std::size_t frames = 0, joints = 0;
        float fps = 30.0F;
        while (head >> word) {
            if (word == "frames") head >> frames;
            else if (word == "fps") head >> fps;
            else if (word == "joints") head >> joints;
        }
        std::vector<std::string> names(lines.begin() + 3, lines.begin() + 3 + static_cast<std::ptrdiff_t>(joints));
        std::vector<std::int32_t> parents;
        std::ifstream pf(dir / "parents.txt");
        for (std::int32_t v; pf >> v;) parents.push_back(v);
        std::ifstream tf(dir / "transforms.f32", std::ios::binary);
        std::vector<float> xf(frames * joints * 12U);
        tf.read(reinterpret_cast<char *>(xf.data()), static_cast<std::streamsize>(xf.size() * sizeof(float)));
        if (!tf) { std::cerr << "transforms.f32 is short\n"; return 1; }
        auto generated = skintokens::soma30_motion_from_transforms(names, parents, fps, xf);
        if (!generated) { std::cerr << generated.error().message << '\n'; return 1; }
        generated->name = argv[6];
        const auto kind = std::string_view{argv[4]} == "once" ? skintokens::clip_cut::once : skintokens::clip_cut::loop;
        auto cut = skintokens::cut_clip(*generated, kind, std::stof(argv[5]));
        if (!cut) { std::cerr << cut.error().message << '\n'; return 1; }
        const auto & r = cut->second;
        std::cout << std::fixed << std::setprecision(3) << "cut " << argv[6] << ": start " << r.start
                  << ", " << r.frames << " frames (" << static_cast<float>(r.frames) / fps << " s), seam "
                  << r.seam_before_max_degrees << " -> " << r.seam_after_max_degrees << " deg, largest step "
                  << r.largest_step_degrees << " deg, travel removed " << r.travel_removed_metres_per_second << " m/s\n";
        std::ifstream rf(argv[2], std::ios::binary);
        const std::vector<char> raw{std::istreambuf_iterator<char>(rf), {}};
        skintokens::humanoid_match_options match_options;
        skintokens::retarget_options transfer_options;
        match_options.minimum_confidence = transfer_options.minimum_confidence = 0.5F;
        auto added = skintokens::add_soma30_motion_glb_memory(
            std::span<const std::byte>{reinterpret_cast<const std::byte *>(raw.data()), raw.size()},
            cut->first, match_options, transfer_options);
        if (!added) { std::cerr << added.error().message << '\n'; return 1; }
        std::ofstream out(argv[7], std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(added->first.data()), static_cast<std::streamsize>(added->first.size()));
        std::cout << "added " << argv[6] << " to " << argv[7] << ": " << added->first.size()
                  << " bytes, humanoid confidence " << added->second.confidence << '\n';
        return out ? 0 : 1;
    }
    if (std::string_view{argv[1]} == "retarget-generated-check") {
        if (argc != 4) { usage(); return 2; }
        auto source = skintokens::load_kimodo_glb_file(argv[2]);
        auto target = skintokens::load_kimodo_glb_file(argv[3]);
        if (!source) { std::cerr << source.error().message << '\n'; return 1; }
        if (!target) { std::cerr << target.error().message << '\n'; return 1; }
        auto mapping = skintokens::match_generated_humanoid(target->rig);
        if (!mapping) { std::cerr << mapping.error().message << '\n'; return 1; }
        auto report = skintokens::compare_soma30_to_generated(*source, *target, *mapping);
        if (!report) { std::cerr << report.error().message << '\n'; return 1; }
        constexpr std::array<std::string_view,30> names{{
            "Hips","Spine1","Spine2","Chest","Neck1","Neck2","Head","Jaw","LeftEye","RightEye",
            "LeftShoulder","LeftArm","LeftForeArm","LeftHand","LeftHandThumbEnd","LeftHandMiddleEnd",
            "RightShoulder","RightArm","RightForeArm","RightHand","RightHandThumbEnd","RightHandMiddleEnd",
            "LeftLeg","LeftShin","LeftFoot","LeftToeBase","RightLeg","RightShin","RightFoot","RightToeBase"}};
        std::cout << std::fixed << std::setprecision(7)
                  << "mean normalized rest-relative displacement error: "
                  << report->mean_normalized_displacement_error << '\n'
                  << "maximum normalized rest-relative displacement error: "
                  << report->maximum_normalized_displacement_error << '\n'
                  << "worst frame: " << report->worst_frame << '\n'
                  << "worst joint: " << names[report->worst_soma30_joint]
                  << " (" << report->worst_soma30_joint << ")\n";
        for (std::size_t joint=0;joint<names.size();++joint)
            if (mapping->soma_to_generated[joint]>=0)
                std::cout << "joint " << names[joint] << ": "
                          << report->per_joint_maximum[joint] << '\n';
        return 0;
    }
    skintokens::runtime_options runtime;
    skintokens::generation_options generation;
    // The viewer's settings: one beam, always the most likely token. Sampled beams (temperature 1)
    // can grow skeletons the skin codes cannot cover; --beams and --temperature still choose.
    generation.beams = 1U;
    generation.temperature = 0.0F;
    auto fit = skintokens::skeleton_fit::global_similarity;
    bool supplied_skeleton = false;
    std::string_view target_rig = "soma30";
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::string_view{argv[i]} == "--device") runtime.device = parse_device(argv[++i]);
        else if (std::string_view{argv[i]} == "--geometric") generation.geometric_only = true;
        else if (std::string_view{argv[i]} == "--supplied-skeleton") supplied_skeleton = true;
        else if (std::string_view{argv[i]} == "--target-rig") {
            target_rig = argv[++i];
            if (target_rig != "soma30" && target_rig != "mixamo52") {
                std::cerr << "target rig must be soma30 or mixamo52\n";
                return 2;
            }
            supplied_skeleton = true;
        }
        else if (std::string_view{argv[i]} == "--retarget-soma-to-mixamo52") {
            target_rig = "mixamo52";
            supplied_skeleton = true;
        }
        else if (std::string_view{argv[i]} == "--fit") {
            try { fit = parse_fit(argv[++i]); }
            catch (const std::invalid_argument & error) { std::cerr << error.what() << '\n'; return 2; }
        }
        else if (std::string_view{argv[i]} == "--no-fit") fit = skintokens::skeleton_fit::none;
        else if (std::string_view{argv[i]} == "--raw-learned") generation.surface_postprocess = false;
        else if (std::string_view{argv[i]} == "--postprocess") generation.surface_postprocess = true;
        else if (std::string_view{argv[i]} == "--no-robust-transfer") generation.robust_transfer = false;
        else if (std::string_view{argv[i]} == "--beams") generation.beams = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (std::string_view{argv[i]} == "--temperature") generation.temperature = std::stof(argv[++i]);
        else if (std::string_view{argv[i]} == "--max-tokens") generation.max_tokens = std::stoul(argv[++i]);
    }
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--geometric") generation.geometric_only = true;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--no-fit") fit = skintokens::skeleton_fit::none;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--supplied-skeleton") supplied_skeleton = true;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--raw-learned") generation.surface_postprocess = false;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--postprocess") generation.surface_postprocess = true;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--retarget-soma-to-mixamo52") {
        target_rig = "mixamo52";
        supplied_skeleton = true;
    }
    auto model = skintokens::model::load(argv[2], runtime);
    if (!model) {
        std::cerr << model.error().message << '\n';
        return 1;
    }
    if (std::string_view{argv[1]} == "inspect") {
        std::cout << "SkinTokens 0.1 model bundle\nbackend: " << model->backend_name() << '\n';
        return 0;
    }
    if (std::string_view{argv[1]} == "rig") {
        if (argc < 5) { usage(); return 2; }
        const std::filesystem::path mesh_path = argv[3];
        auto geometry = mesh_path.extension() == ".t2mesh" ?
            skintokens::load_trellis_mesh_file(mesh_path) : skintokens::load_glb_file(mesh_path);
        if (!geometry) { std::cerr << geometry.error().message << '\n'; return 1; }
        auto binding = model->rig(*geometry, generation);
        if (!binding) { std::cerr << binding.error().message << '\n'; return 1; }

        // A rigged static GLB still needs one identity pose so ordinary glTF
        // viewers can evaluate the skin immediately. It can later receive
        // animation from any compatible retargeting workflow.
        skintokens::motion rest;
        rest.frames = 1U;
        rest.frames_per_second = 30.0F;
        rest.rig = binding->rig;
        rest.root_translations.assign(1U, rest.rig.rest_positions.front());
        rest.local_rotations.assign(rest.rig.names.size(), {});
        // A GLB source keeps its materials, textures and UVs; a T2MESH has none to keep.
        if (mesh_path.extension() != ".t2mesh") {
            std::ifstream stream(mesh_path, std::ios::binary);
            const std::vector<char> raw{std::istreambuf_iterator<char>(stream), {}};
            const std::span<const std::byte> source{reinterpret_cast<const std::byte *>(raw.data()), raw.size()};
            auto bytes = skintokens::save_rigged_source_glb_memory(source, *binding, rest);
            if (bytes) {
                std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
                output.write(reinterpret_cast<const char *>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
                if (!output) { std::cerr << "failed while writing " << argv[4] << '\n'; return 1; }
                std::cout << "learned skeleton and skin weights written to " << argv[4] << '\n';
                return 0;
            }
            std::cerr << "source GLB kept no materials: " << bytes.error().message << '\n';
        }
        auto saved = skintokens::save_skinned_animation_glb_file(
            argv[4], *geometry, *binding, rest);
        if (!saved) { std::cerr << saved.error().message << '\n'; return 1; }
        std::cout << "learned skeleton and skin weights written to " << argv[4] << '\n';
        return 0;
    }
    const bool skin_only = std::string_view{argv[1]} == "skin";
    if ((!skin_only && std::string_view{argv[1]} != "bind") || argc < 6) {
        usage();
        return 2;
    }
    const std::filesystem::path mesh_path = argv[3];
    auto geometry = mesh_path.extension() == ".t2mesh" ?
        skintokens::load_trellis_mesh_file(mesh_path) : skintokens::load_glb_file(mesh_path);
    auto animation = skin_only ? skintokens::load_skeleton_glb_file(argv[4]) :
                                 skintokens::load_kimodo_glb_file(argv[4]);
    if (!geometry) { std::cerr << geometry.error().message << '\n'; return 1; }
    if (!animation) { std::cerr << animation.error().message << '\n'; return 1; }
    if (generation.geometric_only || supplied_skeleton || skin_only) {
      if (fit != skintokens::skeleton_fit::none) {
        auto fitted = skintokens::fit_motion_to_mesh(*geometry, *animation, fit);
        if (!fitted) { std::cerr << fitted.error().message << '\n'; return 1; }
        animation = std::move(fitted);
      }
    }
    if (target_rig == "mixamo52") {
        if (skintokens::identify_rig(animation->rig) != skintokens::rig_kind::soma30) {
            std::cerr << "Mixamo52 retargeting requires a detected SOMA30 skeleton\n";
            return 1;
        }
        auto mixamo = skintokens::make_mixamo52_rig(animation->rig);
        if (!mixamo) { std::cerr << mixamo.error().message << '\n'; return 1; }
        auto report = skintokens::validate_soma30_to_mixamo52(*animation, *mixamo);
        if (!report) { std::cerr << report.error().message << '\n'; return 1; }
        std::cerr << "SOMA30 -> Mixamo52: " << report->isolated_cases
                  << " isolated probes, mean/max " << report->isolated_mean_position_error
                  << '/' << report->isolated_max_position_error
                  << "; clip mean/max " << report->motion_mean_position_error
                  << '/' << report->motion_max_position_error
                  << "; foot velocity relative error " << report->foot_velocity_relative_error << '\n';
        auto retargeted = skintokens::retarget_soma30_to_mixamo52(*animation, *mixamo);
        if (!retargeted) { std::cerr << retargeted.error().message << '\n'; return 1; }
        animation = std::move(retargeted);
    }
    skintokens::result<skintokens::skin> binding = (generation.geometric_only || supplied_skeleton || skin_only) ?
        model->bind(*geometry, animation->rig, generation) : model->rig(*geometry, generation);
    if (!binding) { std::cerr << binding.error().message << '\n'; return 1; }
    if (!generation.geometric_only && !supplied_skeleton && !skin_only) {
        auto matched = skintokens::match_generated_humanoid(binding->rig);
        if (!matched) { std::cerr << matched.error().message << '\n'; return 1; }
        std::cerr << "generated humanoid match: confidence " << matched->report.confidence
                  << ", " << matched->report.mapped_core_joints << " core joints, "
                  << matched->report.ignored_terminal_joints << " flexible hand joints\n";
        auto retargeted = skintokens::retarget_soma30_motion_to_generated(
            *animation, binding->rig, *matched);
        if (!retargeted) { std::cerr << retargeted.error().message << '\n'; return 1; }
        animation = std::move(retargeted);
    }
    auto saved = skintokens::save_skinned_animation_glb_file(argv[5], *geometry, *binding, *animation);
    if (!saved) { std::cerr << saved.error().message << '\n'; return 1; }
    std::cout << (binding->learned ? "learned" : "geometric") << " binding written to " << argv[5] << '\n';
    return 0;
}
