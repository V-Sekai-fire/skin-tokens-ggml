import sys
import numpy as np
exec(open('rest_compare.py').read().split('for a,c in zip')[0])
# Mean z of the skinned mesh in front of / behind the head bone, and the toes' z vs the feet.
for p in sys.argv[1:]:
    g, b = load(p)
    v, _ = skinned(p)
    N = g['nodes']; par = {c: i for i, n in enumerate(N) for c in n.get('children', [])}
    def W(i):
        n = N[i]; M = np.eye(4); M[:3, :3] = q2m(n.get('rotation', [0, 0, 0, 1])) @ np.diag(n.get('scale', [1, 1, 1])); M[:3, 3] = n.get('translation', [0, 0, 0])
        return W(par[i]) @ M if i in par else M
    hb = {k: v['node'] for k, v in g['extensions']['VRMC_vrm']['humanoid']['humanBones'].items()}
    head = W(hb['head'])[:3, 3]
    near = v[np.abs(v[:, 1] - head[1] - 0.05) < 0.05]
    toe = (W(hb['leftToes'])[:3, 3] - W(hb['leftFoot'])[:3, 3]) if 'leftToes' in hb else None
    print(p.split('/')[-1], 'head-height verts: z from', round(float(near[:, 2].min() - head[2]), 3), 'to', round(float(near[:, 2].max() - head[2]), 3),
          '| foot verts mean z - ankle z', round(float(v[v[:, 1] < 0.05][:, 2].mean() - W(hb['leftFoot'])[2, 3]), 3), '| toes dz', None if toe is None else round(float(toe[2]), 3))
