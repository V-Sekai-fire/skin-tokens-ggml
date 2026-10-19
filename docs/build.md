# Build and install

Install Git, CMake 3.25 or newer, Ninja, a C++23 compiler,
`nlohmann-json`, and Vulkan headers, loader, and shader compiler. Vulkan can be
disabled with `-DSKINTOKENS_ENABLE_VULKAN=OFF`. From a source checkout:

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
cmake --install build/release --prefix ./dist
```

The `ggml/` directory holds an upstream GGML tree. During CMake
configuration, the project copies it into the build directory and applies the
ordered compatibility patches in `patches/ggml/`; the `ggml/` tree is
never modified. Reconfiguration reuses the prepared copy while the upstream
revision and patch hashes are unchanged. Advanced builds using a GGML tree that
already contains equivalent fixes can pass
`-DSKINTOKENS_APPLY_GGML_PATCHES=OFF`, and `-DSKINTOKENS_GGML_SOURCE_DIR`
points the build at another GGML tree.

The install contains the shared library, C and C++ headers, CLI, GGML runtime,
and enabled dynamic backends. To test from the build tree:

```sh
ctest --test-dir build/release --output-on-failure
./build/release/bin/skintokens-cli inspect models/SkinTokens-GGUF/F16
```

Nix is optional. It supplies the build tools and Vulkan development packages,
but does not replace the normal CMake build:

```sh
nix develop
cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug
```

The sanitizer build is the normal development lane:

```sh
nix develop
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan -j
ctest --preset asan-ubsan
```
