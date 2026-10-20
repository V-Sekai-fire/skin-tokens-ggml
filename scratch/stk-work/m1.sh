#!/bin/bash
# M1: native SkinTokens reference on the Mac (weights + CLI). Logs to m1.log.
set -euo pipefail
W="$(cd "$(dirname "$0")" && pwd)"
S="$W/../stk"
cd "$W"
echo "STEP env"
if [ ! -f pixi.toml ]; then
  pixi init -q . >/dev/null
  pixi add -q python=3.11 numpy safetensors omegaconf huggingface_hub cmake ninja >/dev/null
  pixi add -q --pypi torch >/dev/null
fi
echo "STEP download"
mkdir -p ckpt
pixi run hf download VAST-AI/SkinTokens \
  experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt \
  experiments/skin_vae_2_10_32768/last.ckpt \
  --revision 79736cad0fd84de384d5eede659b4ebd24effe33 --local-dir ckpt/.download >/dev/null
cp -c ckpt/.download/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt ckpt/grpo_1400.ckpt 2>/dev/null || cp ckpt/.download/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt ckpt/grpo_1400.ckpt
cp -c ckpt/.download/experiments/skin_vae_2_10_32768/last.ckpt ckpt/last.ckpt 2>/dev/null || cp ckpt/.download/experiments/skin_vae_2_10_32768/last.ckpt ckpt/last.ckpt
printf '%s  %s\n' \
  f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692 ckpt/grpo_1400.ckpt \
  4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4 ckpt/last.ckpt | shasum -a 256 -c
echo "STEP build"
cmake -S "$S" -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSKINTOKENS_ENABLE_VULKAN=OFF > build.log 2>&1
cmake --build build -j > build2.log 2>&1
ls -la build/bin | head
echo "M1-DONE"
