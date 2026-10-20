import sys
import numpy as np
exec(open('frame_check.py').read().split('for p in sys.argv')[0])
def joints(p):
    g, b = load(p); N = g['nodes']; par = {c: i for i, n in enumerate(N) for c in n.get('children', [])}
    def W(i):
        n = N[i]; M = np.eye(4)
        if 'matrix' in n: M = np.array(n['matrix']).reshape(4, 4).T
        else: M[:3, :3] = q2m(n.get('rotation', [0, 0, 0, 1])) @ np.diag(n.get('scale', [1, 1, 1])); M[:3, 3] = n.get('translation', [0, 0, 0])
        return W(par[i]) @ M if i in par else M
    hb = g['extensions']['VRMC_vrm']['humanoid']['humanBones']
    return {k: W(v['node'])[:3, 3] for k, v in hb.items()}
a, c = joints(sys.argv[1]), joints(sys.argv[2])
for k in ['hips', 'head', 'leftHand', 'leftFoot', 'leftUpperArm']:
    if k in a and k in c: print(k, 'source', a[k].round(3).tolist(), 'result', c[k].round(3).tolist())
