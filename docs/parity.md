# Reference parity

## Measured parity

Safe checkpoint extraction, checked GGUF loading, CPU/Vulkan backend selection,
the complete TokenRig grammar and Qwen3 stack, Kimodo animation import, and
skinned GLB export are implemented. The F32 Qwen stack reaches `8.1e-6`
relative L2 on CPU; Vulkan reaches `7.1e-4` for the final hidden state and
`1.4e-3` for logits.

The learned binding path runs the released Michelangelo mesh encoder, Qwen
TokenRig policy, FSQ code expansion, condition encoder, and chunked SkinVAE
decoder through GGML. It samples the same 54,000 surface points as upstream
and caps dense decoder queries at 16K vertices per graph. The `skin` command
uses a supplied static or animated skeleton. The `rig` command enables
upstream's unconstrained skeleton generation, which remains experimental
because the released policy can produce unsuitable topologies. `--geometric`
is an explicit non-learned diagnostic.

Upstream's existing-skeleton interface is deliberately generic: `--use_skeleton`
retains an armature found in the input and generates skin only. It does not
promise that every arbitrary hierarchy is in-distribution or prescribe a
retargeting method. Imported GLBs are tokenized as generic articulations. The
released checkpoint stores `order_config`, while `TokenizerPart.parse` reads
`order`, so its executable imported-GLB path emits no spring/part token and
does not activate a hidden Mixamo-specific inference mode from joint names.

The full 54K-point Michelangelo encoder matches the corrected F32 PyTorch
reference at `6.5e-4` relative L2 on Vulkan. A constrained 120-step fixture
checks the production Qwen prefix and greedy policy decision at every step;
Vulkan reproduces every reference code.
The full 54K-point SkinVAE and final raw learned binding are checked
independently. CPU reaches `3.6e-6` decoder relative L2, identical top-four
bones on every vertex, and `1.4e-7` animated-vertex relative L2. The
representative Vulkan fixture reaches `9.1e-4` decoder relative L2, with
99.96% of final top-four slots shared and `5.2e-4` animated-vertex relative L2.
The optional `voxel_skin` heuristic is available with `--postprocess`, matching
the upstream demo's opt-in setting rather than being silently enabled.

## Run the parity checks

Capture operation boundaries from the safe checkpoint in the pinned reference
image, then compare every Qwen, mesh-encoder, SkinVAE, and generation boundary.
The unconstrained fixture additionally records the skeleton/skin switch and
upstream's unusual use of model EOS as the final FSQ code:

```sh
docker build -t skintokens-reference:2.7-parity reference
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  --entrypoint python skintokens-reference:2.7-parity \
  /work/reference/capture_qwen_layer.py \
  --weights /work/generated/safe/tokenrig.safetensors \
  --output /work/fixtures/reference-f32 --device cpu --dtype float32

./build/release/bin/skintokens-qwen-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
./build/release/bin/skintokens-mesh-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
./build/release/bin/skintokens-vae-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
./build/release/bin/skintokens-generation-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
./build/release/bin/skintokens-qwen-batch-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
```

The batch checker compares two different token sequences evaluated together
against two serial evaluations, then repeats the comparison after a shared KV
prefill and a cloned two-child beam branch. This guards batching, cache strides,
and beam-slot cloning on both CPU and Vulkan; configure
`SKINTOKENS_QWEN_FIXTURE_DIR` to include it in the opt-in CTest model suite.

Final binding acceptance captures normalized top-four weights and reference
deformed vertices. The full checker runs the production GGML decoder and
skinning evaluator in one process:

```sh
./build/release/bin/skintokens-full-binding-parity \
  models/skintokens-f32 fixtures/animation-parity cpu
./build/release/bin/skintokens-full-binding-parity \
  models/skintokens-f32 fixtures/animation-parity vulkan
```

For the strict Vulkan diagnostic, disable reduced-precision matrix paths:

```sh
GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 \
GGML_VK_DISABLE_COOPMAT2=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 \
  ./build/release/bin/skintokens-qwen-parity \
  models/skintokens-f32 fixtures/reference-f32 vulkan
```

## Upstream scope

SkinTokens is a learned skin-weight representation. TokenRig is the complete
mesh-to-rig system: a Michelangelo point encoder, Qwen3-0.6B causal model, and
conditional FSQ-VAE decoder. This use is therefore within its intended scope,
not an attempt to treat SkinTokens as a motion generator.
