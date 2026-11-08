#!/usr/bin/env python3
"""Rewrite an F32 SkinTokens GGUF component to BF16x3 hi/lo pairs.

Each matrix tensor `name` (F32) becomes two BF16 tensors:
  `name`     = bf16(value)              (hi part)
  `name.lo`  = bf16(value - f32(hi))     (residual/lo part)

This is XLA's ALG_DOT_BF16_BF16_F32_X3 trick: hi+lo reconstructs the
original F32 value far more precisely than plain bf16 truncation, and a
dot product computed as hi.hi + hi.lo + lo.hi (each term bf16xbf16->f32,
summed in f32, dropping the negligible lo.lo term) recovers ~F32-level
matmul accuracy using only native bf16xbf16 hardware -- which ggml-vulkan
already supports unconditionally on coopmat2 devices, unlike F16 weights
with F32 inputs (no such mixed pipeline exists upstream on coopmat2).

Norm/embedding/sensitive tensors stay F32 untouched, matching
convert_to_gguf.py's own "sensitive" set.
"""

import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType

SENSITIVE = ("norm", "embed", "tok", "query", "FSQ")

REQUIRED_STRING_KEYS = (
    "general.architecture",
    "skintokens.component",
    "skintokens.upstream_revision",
    "skintokens.tokenrig_sha256",
    "skintokens.skin_vae_sha256",
)


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.float32, copy=False).view(np.uint32)
    return (u32 >> np.uint32(16)).astype(np.uint16)


def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def split_hi_lo(arr: np.ndarray):
    hi_u16 = f32_to_bf16_u16(arr)
    hi_f32 = bf16_u16_to_f32(hi_u16)
    residual = arr.astype(np.float32) - hi_f32
    lo_u16 = f32_to_bf16_u16(residual)
    return hi_u16, lo_u16


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    reader = GGUFReader(src_path)

    kv = {}
    for key, field in reader.fields.items():
        if field.types and field.types[0] == GGUFValueType.STRING:
            kv[key] = str(bytes(field.parts[field.data[0]]), "utf-8")

    for key in REQUIRED_STRING_KEYS:
        if key not in kv:
            raise SystemExit(f"source GGUF is missing required key: {key}")

    writer = GGUFWriter(dst_path, arch="skintokens", use_temp_file=False)
    writer.add_string("skintokens.component", kv["skintokens.component"])
    writer.add_string(
        "skintokens.upstream_revision", kv["skintokens.upstream_revision"]
    )
    writer.add_string("skintokens.tokenrig_sha256", kv["skintokens.tokenrig_sha256"])
    writer.add_string("skintokens.skin_vae_sha256", kv["skintokens.skin_vae_sha256"])
    writer.add_uint32("skintokens.format_version", 1)

    converted, kept = 0, 0
    for tensor in reader.tensors:
        name = tensor.name
        data = np.asarray(tensor.data)
        is_matrix = data.ndim >= 2 and tensor.tensor_type == GGMLQuantizationType.F32
        sensitive = any(part in name for part in SENSITIVE)
        if is_matrix and not sensitive:
            hi, lo = split_hi_lo(data)
            writer.add_tensor(
                name, hi, raw_shape=data.shape, raw_dtype=GGMLQuantizationType.BF16
            )
            writer.add_tensor(
                name + ".lo",
                lo,
                raw_shape=data.shape,
                raw_dtype=GGMLQuantizationType.BF16,
            )
            converted += 1
        else:
            writer.add_tensor(name, data)
            kept += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(
        f"split {converted} tensors into bf16 hi/lo pairs, kept {kept} as-is -> {dst_path}"
    )


if __name__ == "__main__":
    main()
