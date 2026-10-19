# Optional end-to-end humanoid pipeline: Trellis2 + Kimodo

Trellis2 can generate and remesh a humanoid, Kimodo can generate its motion,
and skin-tokens.cpp can bind that motion hierarchy to the mesh. This is a
concrete full animated-asset pipeline, not a dependency or requirement of
SkinTokens:

```sh
trellis2cpp/build/examples/mesh2glb \
  trellis-mesh.t2mesh trellis-remeshed.glb 2048 \
  --print 0.5 0.0166667 --quad 20000
./build/release/bin/skintokens-cli skin \
  models/SkinTokens-GGUF/F16 \
  trellis-remeshed.glb kimodo-animation.glb character-animated.glb \
  --device vulkan
```

When the supplied skeleton is detected as SOMA30, it can optionally be
retargeted to the 52-joint order published in upstream's
`configs/skeleton/mixamo.yaml` using an explicit semantic motion map:

```sh
./build/release/bin/skintokens-cli skin \
  models/SkinTokens-GGUF/F16 \
  character.glb kimodo-soma30.glb character-mixamo.glb \
  --device vulkan --retarget-soma-to-mixamo52

./build/release/bin/skintokens-cli retarget-check kimodo-soma30.glb
```

`retarget-check` rotates each mapped source joint by +/-30 degrees around all
three axes and compares corresponding FK trajectories in normalized body
space. It also reports full-clip joint-position and foot-velocity errors. This
tests retargeting independently of learned skin weights. Retargeting is off by
default and appears in the demo only after strong SOMA30 detection.

For an unconstrained SkinTokens result, retain its mesh-fitted generated rig
and learned weights and transfer SOMA30 motion onto that hierarchy afterwards:

```sh
./build/release/bin/skintokens-cli rig \
  models/SkinTokens-GGUF/F16 character.glb character-native.glb --device vulkan
./build/release/bin/skintokens-cli retarget-generated \
  character-native.glb kimodo-soma30.glb character-animated.glb

./build/release/bin/skintokens-cli retarget-generated-check \
  kimodo-soma30.glb character-animated.glb
```

This structural recognizer maps the 22 body joints shared with SOMA30 and
tolerates different generated hand subtrees. Unmatched fingers inherit the
hand motion in a neutral local pose by default; `--map-fingers` copies SOMA's
two endpoint tracks across those branches. Low-confidence or non-humanoid
generated skeletons are rejected rather than matched by spatial proximity.
`retarget-generated-check` compares rest-relative joint displacement in body-
height units and reports the mean, maximum, worst frame, worst joint, and
per-joint maxima.

Both `.glb` and trellis2cpp's versioned `.t2mesh` are accepted by the CLI as mesh input.
Skinning should use the manifold-wrapped, quad-remeshed Trellis export, not its dense raw
marching-cubes reconstruction: the latter is millions of triangles with
generation topology that is unsuitable for animation. Direct quad remeshing is
not sufficient for raw reconstruction topology; Alpha Wrap must clean it first.
GLB scene-node transforms and vertex colours are preserved; atlas textures are
not yet round-tripped. The Kimodo rig is fitted to the mesh before
conditioning, while its animation and hierarchy are retained in the output.
SkinVAE predictions are decoded on the upstream 54K cloud and transferred to
source vertices with the same SciPy cKDTree inverse-distance 8-neighbour
interpolation, then selects and renormalizes the greatest four influences for
glTF. With `--postprocess`, the optional upstream heuristic first multiplies
those channels by its `voxel_skin` surface-geodesic matrix. The C++ diagnostic
retains Open3D voxel ordering and SciPy nearest-neighbour tie behaviour.
