# Command-line usage

The `rig` and `skin` commands take the GGUF bundle directory as `MODEL_DIR`; see
[weights.md](weights.md). Run `skintokens-cli` with no arguments for the full list.

## Rig an arbitrary mesh

The standalone path needs only a static `.glb` mesh. It generates the skeleton
and learned skin weights, then stores a one-frame rest pose in the output so it
opens as a conventional skinned glTF asset:

```sh
./build/release/bin/skintokens-cli rig \
  models/SkinTokens-GGUF/F16 \
  character.glb character-rigged.glb \
  --device vulkan
```

The model accepts arbitrary triangle meshes, although—as with any learned
rigging model—results are strongest on shapes resembling its training
distribution. `--postprocess` enables the upstream surface-locality heuristic;
omit it to preserve the raw learned weights exactly.

## Generate skin weights for an existing skeleton

Use `skin` when the armature already exists. A single rigged GLB can provide
both the geometry and static or animated skeleton; its old weights are ignored:

```sh
./build/release/bin/skintokens-cli skin \
  models/SkinTokens-GGUF/F16 \
  character-rigged.glb character-rigged.glb character-reweighted.glb \
  --device vulkan --fit none
```

Alternatively, pass separate mesh and skeleton GLBs. The default `--fit global`
uses a single uniform scale and translation to match vertical extent and centre;
it retains every relative joint position and bone-length ratio. Use `--fit none`
when both files already share coordinates.

`--fit articulated` is an explicitly experimental alternative for recognized
humanoid arm chains. A motion-only skeleton offers two length-exact poses: its
supplied rest pose, and its own first animation frame. This mode measures both
against the mesh surface along the arm bones and keeps whichever actually runs
inside the arms, so a T-pose rig driving a character generated with its arms
lowered adopts the pose the clip already provides. Doing so also makes the bind
pose and frame zero identical, so playback starts without warping the mesh. An
arm chain that still misses is then posed by analytic two-bone inverse
kinematics toward conservative mesh targets, preserving the globally scaled
upper-arm, forearm, and hand lengths exactly. That fallback's invariant follows
the template-skeleton embedding objective in Baran and Popovic's
[Pinocchio](https://www.tonychai.com/072-baran.pdf), but it is deliberately a
small pose stage rather than a copy of Pinocchio's complete LGPL rigging and
weight-generation library. A fuller distance-field embedding implementation can
replace this isolated stage later.

## Devices

Use `--device vulkan`, `--device cpu`, or `--device auto` at runtime.

## Profiling

Set `SKINTOKENS_PROFILE=1` to print opt-in wall-clock timings for preprocessing,
the mesh and SkinVAE encoders, TokenRig generation, binding integration, and
model loading. TokenRig's summary also reports graph count, host/device transfer
volume, transfer time, synchronized graph-compute time, and device-local KV
cache cloning:

```sh
SKINTOKENS_PROFILE=1 ./build/release/bin/skintokens-cli skin \
  models/SkinTokens-GGUF/F16 mesh.glb skeleton.glb result.glb --device vulkan
```


## Memory

TokenRig keeps an F32 K/V cache on the selected backend. Surviving beams retain
their cache slots, while additional children of the same parent are cloned
device-to-device. Cache memory therefore grows with context length and the
configured beam count; lower `--beams` or `--max-tokens` when device memory is
limited.
