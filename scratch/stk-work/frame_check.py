import json,struct,sys
import numpy as np
CT={5126:('f',4),5125:('I',4),5123:('H',2),5121:('B',1)}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
def load(p):
    b=open(p,'rb').read(); n=struct.unpack('<I',b[12:16])[0]; g=json.loads(b[20:20+n]); return g,b[20+n+8:]
def acc(g,bin_,i):
    a=g['accessors'][i]; f,s=CT[a['componentType']]; c=NC[a['type']]; out=np.zeros((a['count'],c))
    if 'bufferView' in a:
        v=g['bufferViews'][a['bufferView']]; st=v.get('byteStride',s*c); o=v.get('byteOffset',0)+a.get('byteOffset',0)
        raw=np.frombuffer(bin_,dtype=np.uint8,count=st*(a['count']-1)+s*c,offset=o)
        out=np.lib.stride_tricks.as_strided(np.frombuffer(raw.tobytes(),dtype=np.dtype(f)),shape=(a['count'],c),strides=(st,s)).astype(float)
    return out
def q2m(q):
    x,y,z,w=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
for p in sys.argv[1:]:
    g,b=load(p); N=g['nodes']; par={c:i for i,n in enumerate(N) for c in n.get('children',[])}
    def W(i):
        n=N[i]; M=np.eye(4)
        if 'matrix' in n: M=np.array(n['matrix']).reshape(4,4).T
        else: M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0])
        return W(par[i])@M if i in par else M
    vol=0; area_n=0; P=[]
    for i,n in enumerate(N):
        if 'mesh' not in n: continue
        M=np.eye(4) if 'skin' in n else W(i)
        for pr in g['meshes'][n['mesh']]['primitives']:
            if pr.get('mode',4)!=4: continue
            v=acc(g,b,pr['attributes']['POSITION']); v=(M[:3,:3]@v.T).T+M[:3,3]; P.append(v)
            idx=acc(g,b,pr['indices']).astype(int).reshape(-1,3) if 'indices' in pr else np.arange(len(v)).reshape(-1,3)
            a,bb,c=v[idx[:,0]],v[idx[:,1]],v[idx[:,2]]
            vol+=np.einsum('ij,ij->i',a,np.cross(bb,c)).sum()/6
            if 'NORMAL' in pr.get('attributes',{}):
                nn=acc(g,b,pr['attributes']['NORMAL']); nn=(np.linalg.inv(M[:3,:3]).T@nn.T).T
                fn=np.cross(bb-a,c-a); vn=nn[idx].sum(1); area_n+=np.einsum('ij,ij->i',fn,vn).clip(-1,1).sum()
    P=np.vstack(P); lo,hi=P.min(0),P.max(0); ext=hi-lo
    up='XYZ'[int(np.argmax(ext))]
    # forward: feet/toe heuristic: lowest 5% of verts, their centroid offset in z vs whole body centroid
    low=P[P[:,1]<lo[1]+0.05*ext[1]]; fz=low[:,2].mean()-P[:,2].mean()
    print(f"{p.split('/')[-1]}: extent x{ext[0]:.2f} y{ext[1]:.2f} z{ext[2]:.2f} tallest={up}; signed volume {'+' if vol>0 else '-'} ({vol:.4f}) -> winding {'CCW outward' if vol>0 else 'CW/inside-out'}; normals agree with winding: {area_n>0}; feet sit {'+Z' if fz>0 else '-Z'} of body centre ({fz:+.3f})")
