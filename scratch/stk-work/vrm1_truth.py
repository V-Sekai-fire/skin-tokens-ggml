import sys
import numpy as np
from scipy.spatial import cKDTree
exec(open('rest_compare.py').read().split('for a,c in zip')[0])
# Is the source posed away from its bind pose? Compare its skinned rest with its stored (bind) vertices.
def raw(p):
    g, b = load(p); out = []
    for n in g['nodes']:
        if 'mesh' in n and 'skin' in n:
            for pr in g['meshes'][n['mesh']]['primitives']:
                out.append(acc(g, b, pr['attributes']['POSITION']))
    return np.vstack(out)
src, got = sys.argv[1], sys.argv[2]
s_skin, _ = skinned(src); s_raw = raw(src); g_skin, _ = skinned(got)
print('source skinned vs its stored vertices: max', round(float(np.linalg.norm(s_skin - s_raw, axis=1).max()), 5))
print('result vs source stored (bind) vertices: max nearest', round(float(cKDTree(s_raw).query(g_skin)[0].max()), 5))
print('result vs source skinned at its node pose: max nearest', round(float(cKDTree(s_skin).query(g_skin)[0].max()), 5))
