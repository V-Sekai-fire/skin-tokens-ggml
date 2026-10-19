# skin-tokens-ggml

A C++ and GGML port of SkinTokens (TokenRig) that makes a skeleton and skin weights for a static mesh and writes a rigged GLB.

## What it is for

Skin weights tell each mesh vertex how much it follows each bone. Without them, a skeleton cannot deform a character. This library takes a static mesh, predicts a skeleton and its vertex weights, and writes a skinned glTF binary (GLB). It can also make weights for a skeleton that you supply. It runs on the CPU or on Vulkan, and it has a command-line tool, a flat C API and a web demo.

The model is [SkinTokens / TokenRig](https://github.com/VAST-AI-Research/SkinTokens). This port loads converted GGUF weights from the Hugging Face repository `LocalAI-io/SkinTokens-GGUF`.

## Build and run

Install Git, CMake, Ninja, a C++23 compiler, `nlohmann-json`, and the Vulkan headers, loader and shader compiler. Then build and get the F16 weights:

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
hf download LocalAI-io/SkinTokens-GGUF --include "F16/*" --local-dir models/SkinTokens-GGUF
```

Rig a mesh:

```sh
./build/release/bin/skintokens-cli rig models/SkinTokens-GGUF/F16 \
  character.glb character-rigged.glb --device vulkan
```

Run `skintokens-cli` with no arguments to list every command. The `docs/` folder has the build options, command usage, the C API, the demo, checkpoint conversion, reference parity and the humanoid motion pipeline.

## Licence

This project is Apache-2.0; see `LICENSE`. The upstream SkinTokens code and weights are MIT. `NOTICE` keeps the upstream attribution, and `CITATION.cff` lists the sources to cite.
