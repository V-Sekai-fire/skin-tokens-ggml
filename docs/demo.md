# Demo

The dependency-free Go/WebGL demo is upload-first. Choose **Skeleton + skin**
for an unrigged mesh, or **Skin weights only** for either one rigged GLB or
separate mesh and skeleton GLBs. Kimodo animation GLBs work directly as the
separate skeleton input. Skeleton + skin also accepts an optional Kimodo
SOMA30 animation: the demo keeps the generated native rig, transfers the
motion onto it, and offers both files for download. Playback appears only when the supplied hierarchy is
animated. Separate inputs default to the length-preserving global fit and offer
articulated fitting as an experimental checkbox. SOMA30 retargeting appears
only when that rig is detected. The demo keeps persistent history, runs one
bounded worker, visualizes joint influences, and exports reusable GLBs. For a
generated rig driven by SOMA30, the viewer can overlay the normalized source
motion in blue, draw the retarget error vectors, and colour mapped joints by
divergence. Upload selections are retained when switching workflows: an
unrigged Skeleton + skin mesh becomes the separate character mesh in Skin
weights only, while a rigged input becomes the one-file asset. The optional
driving motion has an explicit Clear button.

```sh
cd demo
go run . \
  --listen 127.0.0.1:8095 \
  --cli ../build/release/bin/skintokens-cli \
  --model ../models/SkinTokens-GGUF/F16
```

Open `http://localhost:8095`. Paths and the listen address are flags; no local
machine paths are compiled into the C++ library or web server. Optional
giraffe-example paths default to disabled.
