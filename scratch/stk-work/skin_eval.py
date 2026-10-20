import sys,json,struct
import numpy as np
exec(open('frame_check.py').read().split('for p in sys.argv')[0])
def evaluate(p, frame, clip_name=None):
    g,b=load(p); N=g['nodes']; par={c:i for i,n in enumerate(N) for c in n.get('children',[])}
    T=[np.array(n.get('translation',[0,0,0]),float) for n in N]; Q=[np.array(n.get('rotation',[0,0,0,1]),float) for n in N]
    if clip_name is not None:
        clip=[a for a in g['animations'] if a.get('name')==clip_name][0]
        for ch in clip['channels']:
            s=clip['samplers'][ch['sampler']]; out=acc(g,b,s['output']); k=min(frame,len(out)-1)
            if ch['target']['path']=='rotation': Q[ch['target']['node']]=out[k]
            elif ch['target']['path']=='translation': T[ch['target']['node']]=out[k]
    W={}
    def w(i):
        if i in W: return W[i]
        M=np.eye(4); M[:3,:3]=q2m(Q[i])@np.diag(N[i].get('scale',[1,1,1])); M[:3,3]=T[i]
        W[i]=w(par[i])@M if i in par else M; return W[i]
    sk=g['skins'][0]; J=sk['joints']; IB=acc(g,b,sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
    S=np.array([w(j)@IB[k] for k,j in enumerate(J)])
    out=[]
    for i,n in enumerate(N):
        if 'mesh' not in n or 'skin' not in n: continue
        for pr in g['meshes'][n['mesh']]['primitives']:
            a=pr['attributes']; v=acc(g,b,a['POSITION']); jj=acc(g,b,a['JOINTS_0']).astype(int); ww=acc(g,b,a['WEIGHTS_0'])
            vh=np.c_[v,np.ones(len(v))]; M=np.einsum('vk,vkij->vij',ww,S[jj]); out.append(np.einsum('vij,vj->vi',M,vh)[:,:3])
    return np.vstack(out)
old,new=sys.argv[1],sys.argv[2]
for clip,frames in [('RESET',[0]),('walk',[0,60,200,900])]:
    for f in frames:
        try:
            a=evaluate(old,f,clip); c=evaluate(new,f,clip)
        except IndexError: continue
        d=np.linalg.norm(a-c,axis=1); h=np.ptp(a[:,1])
        print(f"{new.split('/')[-1]} {clip} frame {f}: max vertex difference {d.max():.2e} m ({d.max()/h:.1e} of height), mean {d.mean():.1e}")
