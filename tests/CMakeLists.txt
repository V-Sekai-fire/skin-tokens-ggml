add_executable(skintokens-api-test api_test.cpp)
target_link_libraries(skintokens-api-test PRIVATE skintokens::skintokens)
add_test(NAME skintokens-api-test COMMAND skintokens-api-test)

add_executable(skintokens-c-api-test c_api_test.c)
target_link_libraries(skintokens-c-api-test PRIVATE skintokens::skintokens)
add_test(NAME skintokens-c-api-test COMMAND skintokens-c-api-test)

# Compile the exact production integration source into a small fixture test so
# final learned-joint selection is covered without downloading model weights.
add_executable(skintokens-binding-test binding_test.cpp ../src/binding.cpp)
target_include_directories(skintokens-binding-test PRIVATE ../src ../include)
add_test(NAME skintokens-binding-test COMMAND skintokens-binding-test)

add_executable(skintokens-robust-transfer-test robust_transfer_test.cpp ../src/binding.cpp)
target_include_directories(skintokens-robust-transfer-test PRIVATE ../src ../include)
add_test(NAME skintokens-robust-transfer-test COMMAND skintokens-robust-transfer-test)
add_executable(skintokens-rigid-parts-test rigid_parts_test.cpp ../src/binding.cpp)
target_include_directories(skintokens-rigid-parts-test PRIVATE ../src ../include)
add_test(NAME skintokens-rigid-parts-test COMMAND skintokens-rigid-parts-test)

add_executable(skintokens-tokenizer-test tokenizer_test.cpp ../src/tokenizer.cpp)
target_include_directories(skintokens-tokenizer-test PRIVATE ../src ../include)
target_link_libraries(skintokens-tokenizer-test PRIVATE ggml)
add_test(NAME skintokens-tokenizer-test COMMAND skintokens-tokenizer-test)

add_executable(skintokens-skeleton-grammar-test skeleton_grammar_test.cpp)
target_include_directories(skintokens-skeleton-grammar-test PRIVATE ../src)
add_test(NAME skintokens-skeleton-grammar-test COMMAND skintokens-skeleton-grammar-test)

add_executable(skintokens-source-glb-test source_glb_test.cpp)
target_link_libraries(skintokens-source-glb-test PRIVATE skintokens::skintokens nlohmann_json::nlohmann_json)
add_test(NAME skintokens-source-glb-test COMMAND skintokens-source-glb-test)

add_executable(skintokens-vrm-pose-test vrm_pose_test.cpp)
target_link_libraries(skintokens-vrm-pose-test PRIVATE skintokens::skintokens nlohmann_json::nlohmann_json)
add_test(NAME skintokens-vrm-pose-test COMMAND skintokens-vrm-pose-test)

add_executable(skintokens-motion-clip-test motion_clip_test.cpp)
target_link_libraries(skintokens-motion-clip-test PRIVATE skintokens::skintokens)
add_test(NAME skintokens-motion-clip-test COMMAND skintokens-motion-clip-test)

add_executable(skintokens-non-indexed-glb-test non_indexed_glb_test.cpp)
target_link_libraries(skintokens-non-indexed-glb-test PRIVATE skintokens::skintokens nlohmann_json::nlohmann_json)
add_test(NAME skintokens-non-indexed-glb-test COMMAND skintokens-non-indexed-glb-test)
