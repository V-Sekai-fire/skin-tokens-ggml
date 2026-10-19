# C API

[`skintokens.h`](include/skintokens/skintokens.h) is a flat C11-compatible API.
It uses an opaque model handle, caller-owned fixed error buffers, explicit
defaults, and status returns; C++ exceptions never cross the ABI. Returned
backend and `last_error` strings are borrowed from the model and remain valid
until its next call or destruction.

```c
#include <skintokens/skintokens.h>

char error[512];
st_model *model = NULL;
st_runtime_options runtime = st_default_runtime_options();
runtime.device = ST_DEVICE_VULKAN;
if (st_model_load("models/SkinTokens-GGUF/F16", &runtime, &model,
                  error, sizeof(error)) != ST_OK) {
    /* report error */
}

st_generation_options generation = st_default_generation_options();
generation.surface_postprocess = 1;
int learned = 0;
st_status result = st_skin_files(model, "character.glb", "motion.glb",
                                 "animated.glb", ST_FIT_GLOBAL_SIMILARITY,
                                 &generation, &learned,
                                 error, sizeof(error));
st_model_free(model);
```

The generated-rig retarget surface uses opaque option and match handles too,
so PureGo and other FFI callers never have to predict a C structure layout:

```c
st_retarget_options *options = NULL;
st_humanoid_match *match = NULL;
st_retarget_options_create(&options, error, sizeof(error));
st_retarget_options_set_minimum_confidence(options, 0.85f,
                                           error, sizeof(error));
st_humanoid_match_glb_file("character-native.glb", options, &match,
                           error, sizeof(error));
st_retarget_soma30_glb_files("character-native.glb", "motion.glb",
                             "character-animated.glb", match, options,
                             error, sizeof(error));
st_humanoid_match_free(match);
st_retarget_options_free(options);
```

`st_rig_file` generates a skeleton and weights; `st_skin_files` generates
weights for a supplied static or animated skeleton. Pass the same GLB as both
mesh and skeleton paths for a single rigged asset. `st_inspect_glb_file`
reports mesh, skin, skeleton, animation, and recognized-rig metadata without
loading GGUF weights. The public parsing and error-buffer surface is exercised
by the ASan/UBSan libFuzzer build; GGUF loading is intentionally kept outside
that fuzzer.
