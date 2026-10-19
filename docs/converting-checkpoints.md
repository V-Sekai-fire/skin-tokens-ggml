# Convert the upstream checkpoints yourself

This is an advanced reproducibility path; most users should download the GGUF
bundle above. The two official checkpoints are MIT-labelled but use Lightning
pickle containers. Downloads are explicit, hash-checked, and excluded from
git:

```sh
./scripts/download_models.sh checkpoints
docker build -t skintokens-reference:2.7 reference
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  skintokens-reference:2.7 \
  --tokenrig /work/checkpoints/grpo_1400.ckpt \
  --skin-vae /work/checkpoints/last.ckpt \
  --output /work/generated/safe
python scripts/convert_to_gguf.py \
  --tokenrig generated/safe/tokenrig.safetensors \
  --skin-vae generated/safe/skin-vae.safetensors \
  --config generated/safe/config.json \
  --output models/skintokens-f32 --ftype 0
```

The normal converter accepts safetensors only. Every GGUF component records
the upstream revision and both intermediate SHA-256 identities, and model load
rejects mixed bundles.
