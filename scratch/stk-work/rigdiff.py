"""Compare two rigged GLBs from the same mesh: joint count, joint positions, dominant joint per vertex.

    python rigdiff.py A.glb B.glb
Joint positions are world-space rest positions (node translations composed down the skin's
hierarchy). Distances are printed in the mesh's units and as a fraction of the mesh height.
"""
import json
import struct
import sys

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    doc = json.loads(b[20:20 + jl])
    binary = b[20 + jl + 8:]

    def acc(i):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        n = WIDTH[a["type"]]
        dt = np.dtype(COMP[a["componentType"]])
        off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride", 0) or n * dt.itemsize
        raw = np.frombuffer(binary, np.uint8, count=stride * (a["count"] - 1) + n * dt.itemsize, offset=off)
        out = np.empty((a["count"], n), dt)
        for k in range(a["count"]):
            out[k] = np.frombuffer(raw[k * stride:k * stride + n * dt.itemsize].tobytes(), dt)
        if a.get("normalized") and dt != np.float32:
            out = out.astype(np.float32) / np.iinfo(dt).max
        return out

    skin = doc["skins"][0]
    joints = skin["joints"]
    parent = {}
    for i, nd in enumerate(doc["nodes"]):
        for c in nd.get("children", []):
            parent[c] = i

    def world(i):
        t = np.array(doc["nodes"][i].get("translation", [0, 0, 0]), float)
        return t + (world(parent[i]) if i in parent else 0)

    pos = np.array([world(j) for j in joints])
    prim = doc["meshes"][0]["primitives"][0]["attributes"]
    verts = acc(prim["POSITION"]).astype(float)
    jt = acc(prim["JOINTS_0"]).astype(int)
    wt = acc(prim["WEIGHTS_0"]).astype(float)
    dom = jt[np.arange(len(jt)), wt.argmax(1)]
    return pos, verts, dom


def main():
    pa, va, da = load(sys.argv[1])
    pb, vb, db = load(sys.argv[2])
    h = va[:, 1].max() - va[:, 1].min()
    print(f"joints {len(pa)} vs {len(pb)}; mesh height {h:.3f}")
    if len(pa) == len(pb):
        d = np.linalg.norm(pa - pb, axis=1)
        print(f"joint distance: max {d.max():.4f} ({d.max() / h:.2%} of height), mean {d.mean():.4f}")
        print(f"dominant joint agrees on {np.mean(da == db):.2%} of {len(da)} vertices")
    # Nearest-joint match works across different joint counts.
    dd = np.linalg.norm(pa[:, None] - pb[None], axis=2)
    print(f"each A joint to nearest B joint: max {dd.min(1).max() / h:.2%}, mean {dd.min(1).mean() / h:.2%} of height")
    print(f"each B joint to nearest A joint: max {dd.min(0).max() / h:.2%}, mean {dd.min(0).mean() / h:.2%} of height")


main()
