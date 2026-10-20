import sys
import numpy as np
exec(open('rest_compare.py').read().split('for a,c in zip')[0])
# The original VRM 0.x mesh (faces -Z), turned 180 degrees about Y by hand, is the expected VRM 1.0 mesh.
orig, _ = skinned(sys.argv[1])
want = orig * np.array([-1, 1, -1])
for p in sys.argv[2:]:
    got, _ = skinned(p)
    from scipy.spatial import cKDTree
    def nearest(a, b):
        return cKDTree(b).query(a)[0]
    print(p.split('/')[-1], 'nearest vertex of the original turned by hand: max', round(float(nearest(got, want).max()), 5), 'm; of the unturned original: max', round(float(nearest(got, orig).max()), 5), 'm')
