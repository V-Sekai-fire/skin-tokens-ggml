import sys
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
exec(open('rest_compare.py').read().split('for a,c in zip')[0])
src, got = sys.argv[1], sys.argv[2]
s_skin, _ = skinned(src)
g, b = load(got)
v, _ = skinned(got)
d = cKDTree(s_skin).query(v)[0]
# main joint of each result vertex
names = []
for n in g['nodes']:
    if 'mesh' in n and 'skin' in n:
        sk = g['skins'][n['skin']]
        for pr in g['meshes'][n['mesh']]['primitives']:
            j = acc(g, b, pr['attributes']['JOINTS_0']).astype(int); w = acc(g, b, pr['attributes']['WEIGHTS_0'])
            top = j[np.arange(len(j)), w.argmax(1)]
            names += [g['nodes'][sk['joints'][t]].get('name') for t in top]
names = np.array(names)
far = d > 0.005
print('vertices moved > 5 mm:', int(far.sum()), 'of', len(d))
print('their main bones:', Counter(names[far]).most_common(8))
