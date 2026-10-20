"""Gate every Hugging Face upload on a scan for blocked words, so the check never depends on recall.

    space_gate.py scan <file or dir>...              exit 1 on any finding
    space_gate.py push <repo> <type> <root> <rel>... scan, commit, then compare live sha256
    space_gate.py audit <repo> <type>                download the live repo and scan all of it
    space_gate.py --self-test                        planted files must fail, a clean one pass

The HF token arrives on stdin (`bao-hf-token | ... push ...`), never on argv. Archives
(.zip, .tar, .tar.gz, .tgz) are opened and their members scanned; other binaries (.pck, .wasm,
.png) are scanned as raw bytes, which is enough because Godot stores pck paths and file
contents uncompressed. The word list lives here, in a private repo, and never in the Space.
"""
import hashlib
import io
import os
import re
import sys
import tarfile
import tempfile
import zipfile

ANYWHERE = [rb"/Users/[a-z.]+", rb"ernest\.lee", rb"chibifire-smoke", rb"\.ts\.net",
            rb"stonecat-ratio"]
PAGE_ONLY = [rb"127\.0\.0\.1", rb"localhost"]
PAGE = (".html", ".htm", ".md")
ANY_RE = re.compile(b"|".join(ANYWHERE), re.I)
PAGE_RE = re.compile(b"|".join(ANYWHERE + PAGE_ONLY), re.I)


def scan_bytes(name, data, out):
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for i in z.infolist():
                scan_name(f"{name}!{i.filename}", out)
                if not i.is_dir():
                    scan_bytes(f"{name}!{i.filename}", z.read(i), out)
        return
    if name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t.getmembers():
                scan_name(f"{name}!{m.name}", out)
                if m.isfile():
                    scan_bytes(f"{name}!{m.name}", t.extractfile(m).read(), out)
        return
    if name.endswith(".parquet"):
        return scan_parquet(name, data, out)
    rx = PAGE_RE if name.lower().endswith(PAGE) else ANY_RE
    for m in rx.finditer(data):
        out.append(f"{name}: {m.group().decode('latin-1')!r} at byte {m.start()}")


def scan_parquet(name, data, out):
    """Parquet pages are compressed, so raw bytes hide strings: decode every string column."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = pq.ParquetFile(io.BytesIO(data) if data is not None else name)
    meta = f.schema_arrow.metadata or {}
    for k, v in meta.items():
        for m in ANY_RE.finditer(k + b"=" + v):
            out.append(f"{name}: metadata {m.group().decode('latin-1')!r}")
    textual = lambda t: pa.types.is_string(t) or pa.types.is_large_string(t) or (
        pa.types.is_list(t) and textual(t.value_type))
    cols = [x.name for x in f.schema_arrow if textual(x.type)]
    for c in cols:
        if ANY_RE.search(c.encode()):
            out.append(f"{name}: column name {c!r}")
    if not cols:
        return
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=cols)
        for c in cols:
            vals = t.column(c).combine_chunks()
            while pa.types.is_list(vals.type):
                vals = vals.flatten()
            for m in ANY_RE.finditer("\n".join(set(x for x in vals.to_pylist() if x)).encode()):
                out.append(f"{name}: column {c} row group {i}: {m.group().decode('latin-1')!r}")


def scan_name(name, out):
    if ANY_RE.search(name.encode()):
        out.append(f"{name}: blocked word in the file name")


def scan(paths):
    out = []
    for p in paths:
        files = [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs] if os.path.isdir(p) else [p]
        for f in sorted(files):
            if "/.git/" in f or f.endswith("/.git"):
                continue
            if f.endswith(".parquet"):
                scan_name(f, out)
                scan_parquet(f, None, out)
                continue
            with open(f, "rb") as fh:
                head = fh.read(64)
            if head.startswith(b"version https://git-lfs"):
                out.append(f"{f}: is an LFS pointer, not the real file; smudge it first")
                continue
            data = open(f, "rb").read()
            scan_name(f, out)
            scan_bytes(f, data, out)
    return out


def push(repo, rtype, root, rels):
    from huggingface_hub import CommitOperationAdd, HfApi
    found = scan([os.path.join(root, r) for r in rels])
    if found:
        print("\n".join(found)); print(f"GATE-FAIL {len(found)} finding(s); nothing uploaded"); return 1
    tok = sys.stdin.read().strip()
    api = HfApi(token=tok)
    print("token len", len(tok), "whoami", api.whoami()["name"])
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=os.path.join(root, r)) for r in rels]
    c = api.create_commit(repo, repo_type=rtype, operations=ops,
                          commit_message=os.environ.get("MSG", "Update " + ", ".join(rels)))
    bad = 0
    for r in rels:
        live = api.hf_hub_download(repo, r, repo_type=rtype, revision=c.oid)
        h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h(live) != h(os.path.join(root, r)):
            bad += 1; print("LIVE-DIFF", r)
    print("commit", c.oid[:7], "live matches" if not bad else f"{bad} live file(s) differ")
    return 1 if bad else 0


def audit(repo, rtype):
    from huggingface_hub import snapshot_download
    tok = sys.stdin.read().strip()
    with tempfile.TemporaryDirectory() as d:
        snapshot_download(repo, repo_type=rtype, local_dir=d, token=tok)
        found = [f.replace(d, repo) for f in scan([d]) if "/.cache/" not in f]
    print("\n".join(found)); print(("AUDIT-FAIL %d finding(s)" % len(found)) if found else "AUDIT-OK")
    return 1 if found else 0


def self_test():
    with tempfile.TemporaryDirectory() as d:
        def w(rel, data):
            p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(data); return p
        clean = [w("ok/index.html", b"<p>SK_Robot and Sophia</p>"), w("ok/index.js", b"host='localhost'")]
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as z:
            z.writestr("app/stonecat-ratio/m.glb", b"glTF")
        tb = io.BytesIO()
        with tarfile.open(fileobj=tb, mode="w:gz") as t:
            info = tarfile.TarInfo("x/readme.txt"); body = b"built in /Users/someone/x"
            info.size = len(body); t.addfile(info, io.BytesIO(body))
        plants = {
            "pck holds a home-folder path": w("p/rig/index.pck", b"GDPC\x00res:///Users/someone/m.glb\x00"),
            "html names a local server": w("p/page.html", b"<a href='http://127.0.0.1:8721/'>"),
            "markdown names a tailnet host": w("p/README.md", b"see fedora.example.ts.net"),
            "zip member name": w("p/dl.zip", zb.getvalue()),
            "tar.gz member content": w("p/dl.tar.gz", tb.getvalue()),
            "file name itself": w("p/ernest.lee-render.png", b"\x89PNG"),
            "LFS pointer instead of the file": w("p/big.wasm", b"version https://git-lfs.github.com/spec/v1\n"),
        }
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            pb = io.BytesIO()
            pq.write_table(pa.table({"uuid": ["a"], "joint_names": [["Hips", "/Users/someone/Head"]]}), pb,
                           compression="zstd")
            plants["parquet list<string> value"] = w("p/skeletons.parquet", pb.getvalue())
            cb = io.BytesIO()
            pq.write_table(pa.table({"uuid": ["a"], "joint_names": [["Hips", "Head"]]}), cb,
                           compression="zstd")
            clean.append(w("ok/skeletons.parquet", cb.getvalue()))
        except ImportError:
            print("SKIP parquet controls: no pyarrow in this interpreter")
        ok = 0
        r = scan(clean)
        print(("ok  " if not r else "BAD ") + "clean files pass (localhost in .js is allowed)"); ok += not r
        for what, p in plants.items():
            r = scan([p])
            print(("ok  " if r else "BAD ") + f"plant fails: {what}"); ok += bool(r)
        n = len(plants) + 1
        print(f"{ok} of {n} control(s) behave as planted")
        return 0 if ok == n else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--self-test"]:
        sys.exit(self_test())
    if a and a[0] == "scan":
        f = scan(a[1:]); print("\n".join(f)); print(f"GATE-FAIL {len(f)}" if f else "GATE-OK"); sys.exit(1 if f else 0)
    if a and a[0] == "push":
        sys.exit(push(a[1], a[2], a[3], a[4:]))
    if a and a[0] == "audit":
        sys.exit(audit(a[1], a[2]))
    print(__doc__); sys.exit(2)
