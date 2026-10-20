import sys,json
import numpy as np
exec(open('frame_check.py').read().split('for p in sys.argv')[0])
def skinned(p):
    g,b=load(p); N=g['nodes']; par={c:i for i,n in enumerate(N) for c in n.get('children',[])}
    W={}
    def w(i):
        if i in W: return W[i]
        n=N[i]; M=np.eye(4)
        if 'matrix' in n: M=np.array(n['matrix']).reshape(4,4).T
        else: M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0])
        W[i]=w(par[i])@M if i in par else M; return W[i]
    out=[]
    for i,n in enumerate(N):
        if 'mesh' not in n: continue
        if 'skin' in n:
            sk=g['skins'][n['skin']]; IB=acc(g,b,sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
            S=np.array([w(j)@IB[k] for k,j in enumerate(sk['joints'])])
        for pr in g['meshes'][n['mesh']]['primitives']:
            a=pr['attributes']; v=acc(g,b,a['POSITION']); vh=np.c_[v,np.ones(len(v))]
            if 'skin' in n and 'JOINTS_0' in a:
                jj=acc(g,b,a['JOINTS_0']).astype(int); ww=acc(g,b,a['WEIGHTS_0'])
                M=np.einsum('vk,vkij->vij',ww,S[jj]); out.append(np.einsum('vij,vj->vi',M,vh)[:,:3])
            else: out.append((w(i)@vh.T).T[:,:3])
    cols=[]
    ext=g.get('extensions',{}).get('VRMC_springBone',{})
    for c in ext.get('colliders',[]):
        for k,sh in c['shape'].items():
            for f in ('offset','tail'):
                if f in sh: cols.append((w(c['node'])@np.r_[sh[f],1])[:3])
    la=g.get('extensions',{}).get('VRMC_vrm',{}).get('lookAt',{}).get('offsetFromHeadBone')
    if la: cols.append((w(g['extensions']['VRMC_vrm']['humanoid']['humanBones']['head']['node'])@np.r_[la,1])[:3])
    return np.vstack(out), np.array(cols)
for a,c in zip(sys.argv[1::2],sys.argv[2::2]):
    va,ca=skinned(a); vc,cc=skinned(c)
    d=np.linalg.norm(va-vc,axis=1); h=np.ptp(va[:,1])
    cd=np.linalg.norm(ca-cc,axis=1).max() if len(ca) else 0
    print(f"{c.split('/')[-1]}: {len(va)} vertices, max move {d.max():.2e} m; {len(ca)} collider/look-at points, max move {cd:.2e} m")
