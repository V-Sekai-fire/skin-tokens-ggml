import sys
import numpy as np
exec(open('rest_compare.py').read().split('for a,c in zip')[0])
a, _ = skinned(sys.argv[1]); c, _ = skinned(sys.argv[2])
d = np.linalg.norm(a - c, axis=1)
print('moved >1cm', int((d > 0.01).sum()), 'of', len(d))
idx = np.argsort(-d)[:4]
print('A', a[idx].round(3).tolist()); print('B', c[idx].round(3).tolist())
g1, b1 = load(sys.argv[1]); g2, b2 = load(sys.argv[2])
for k, z in (('translation', [0, 0, 0]), ('rotation', [0, 0, 0, 1])):
    diffs = [i for i, (x, y) in enumerate(zip(g1['nodes'], g2['nodes'])) if np.abs(np.array(x.get(k, z)) - np.array(y.get(k, z))).max() > 1e-4]
    print(k, 'nodes differing', len(diffs), [g1['nodes'][i].get('name') for i in diffs[:8]])
