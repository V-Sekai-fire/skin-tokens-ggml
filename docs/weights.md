# Weights

Downloading the converted F16 GGUF bundle is the recommended setup. With the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf download LocalAI-io/SkinTokens-GGUF \
  --include "F16/*" \
  --local-dir models/SkinTokens-GGUF
```

Use `models/SkinTokens-GGUF/F16` as `MODEL_DIR` in the commands below. The
repository also contains an F32 bundle for numerical parity work; normal
inference should use F16.

To make the GGUF bundle from the upstream checkpoints, see
[converting-checkpoints.md](converting-checkpoints.md).
