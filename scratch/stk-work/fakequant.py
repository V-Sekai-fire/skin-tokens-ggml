"""Fake-quantize the matrices of a BF16 GGUF in place (the file stays BF16).

Usage: python fakequant.py <in.gguf> <out.gguf> <mode> [--prefix llm.] [--keep llm.tok.w,llm.out.w]

Modes:
  identity   write every value back unchanged (control: the rig must stay byte-identical)
  t158       ternary, absmean scale per group of 128 inputs (BitNet b1.58 style)
  t158h      ternary in a Hadamard-rotated basis: W -> Q(W R) R^T, R = diag(signs) * WHT128,
             which equals rotating the activations at run time (Bonsai 2 style)
  q4         4-bit symmetric, absmax scale per group of 32 inputs (Q4_0-like, for reference)

Only 2-D BF16 tensors whose name starts with --prefix and is not in --keep change.
"""
import struct, sys, shutil, argparse
import numpy as np

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BF16 = 30


def rd(f, fmt):
    return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))


def rstr(f):
    n, = rd(f, 'Q')
    return f.read(n).decode('utf-8', 'replace')


def rval(f, t):
    if t == 8:
        return rstr(f)
    if t == 9:
        et, n = rd(f, 'IQ')
        return [rval(f, et) for _ in range(n)]
    b = f.read(SZ[t])
    return struct.unpack('<' + {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}[t], b)[0]


def tensors(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        _, nt, nkv = rd(f, 'IQQ')
        align = 32
        for _ in range(nkv):
            k = rstr(f)
            t, = rd(f, 'I')
            v = rval(f, t)
            if k == 'general.alignment':
                align = v
        out = []
        for _ in range(nt):
            name = rstr(f)
            nd, = rd(f, 'I')
            shape = rd(f, 'Q' * nd)
            ty, off = rd(f, 'IQ')
            out.append((name, shape, ty, off))
        base = (f.tell() + align - 1) // align * align
    return base, out


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def wht(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def quant(w, mode, rng):
    # w: (rows, cols) float32, cols = inputs (ggml ne0, contiguous)
    rows, cols = w.shape
    if mode == 'identity':
        return w
    if mode == 'q4':
        g = w.reshape(rows, cols // 32, 32)
        s = np.abs(g).max(-1, keepdims=True) / 7.0
        s[s == 0] = 1
        return (np.clip(np.round(g / s), -8, 7) * s).reshape(rows, cols)
    gs = 128
    assert cols % gs == 0, (rows, cols)
    if mode == 't158h':
        signs = rng.choice([-1.0, 1.0], size=cols)
        r = wht(gs)
        g = (w * signs).reshape(rows, cols // gs, gs) @ r
    else:
        g = w.reshape(rows, cols // gs, gs)
    s = np.abs(g).mean(-1, keepdims=True)
    s[s == 0] = 1
    q = np.clip(np.round(g / s), -1, 1) * s
    if mode == 't158h':
        q = (q @ r.T).reshape(rows, cols) * signs
    return q.reshape(rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('mode')
    ap.add_argument('--prefix', default='llm.')
    ap.add_argument('--keep', default='llm.tok.w,llm.out.w')
    a = ap.parse_args()
    keep = set(a.keep.split(',')) if a.keep else set()
    rng = np.random.default_rng(0)
    shutil.copyfile(a.src, a.dst)
    base, ts = tensors(a.src)
    mm = np.memmap(a.dst, dtype=np.uint8, mode='r+')
    n_changed = n_params = 0
    err = []
    for name, shape, ty, off in ts:
        if ty != BF16 or len(shape) != 2 or not name.startswith(a.prefix) or name in keep:
            continue
        cols, rows = shape  # ggml order: ne0 = inputs
        nb = rows * cols * 2
        raw = mm[base + off: base + off + nb].view(np.uint16).reshape(rows, cols)
        w = bf16_to_f32(raw)
        q = quant(w, a.mode, rng)
        newraw = f32_to_bf16(q)
        mm[base + off: base + off + nb] = newraw.view(np.uint8).reshape(-1)
        n_changed += 1
        n_params += rows * cols
        err.append(float(np.linalg.norm(bf16_to_f32(newraw) - w) / (np.linalg.norm(w) + 1e-12)))
    mm.flush()
    print(f'{a.mode}: changed {n_changed} tensors, {n_params / 1e6:.1f}M weights, '
          f'relative error mean {np.mean(err) if err else 0:.4f} max {max(err) if err else 0:.4f}')


if __name__ == '__main__':
    main()
