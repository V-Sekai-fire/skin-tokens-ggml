import sys,json,math
import numpy as np
exec(open('frame_check.py').read().split('for p in sys.argv')[0])
def worlds(g):
    N=g['nodes']; par={c:i for i,n in enumerate(N) for c in n.get('children',[])}; W={}
    def w(i):
        if i in W: return W[i]
        n=N[i]; M=np.eye(4)
        if 'matrix' in n: M=np.array(n['matrix']).reshape(4,4).T
        else: M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0])
        W[i]=w(par[i])@M if i in par else M; return W[i]
    return [w(i) for i in range(len(N))]
def verts(g,b,Wj):
    out=[]
    for i,n in enumerate(g['nodes']):
        if 'mesh' not in n or 'skin' not in n: continue
        sk=g['skins'][n['skin']]; IB=acc(g,b,sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        S=np.array([Wj[j]@IB[k] for k,j in enumerate(sk['joints'])])
        for pr in g['meshes'][n['mesh']]['primitives']:
            a=pr['attributes']
            if 'JOINTS_0' not in a: continue
            v=acc(g,b,a['POSITION']); jj=acc(g,b,a['JOINTS_0']).astype(int); ww=acc(g,b,a['WEIGHTS_0'])
            M=np.einsum('vk,vkij->vij',ww,S[jj]); out.append(np.einsum('vij,vj->vi',M,np.c_[v,np.ones(len(v))])[:,:3])
    return np.vstack(out)
for old,new in zip(sys.argv[1::2],sys.argv[2::2]):
    go,bo=load(old); gn,bn=load(new); Wn=worlds(gn)
    hb={k:v['node'] for k,v in gn['extensions']['VRMC_vrm']['humanoid']['humanBones'].items()}
    worst=0; worst_name=''
    for a,c,t in [('leftUpperArm','leftLowerArm',[1,0,0]),('leftLowerArm','leftHand',[1,0,0]),('rightUpperArm','rightLowerArm',[-1,0,0]),('rightLowerArm','rightHand',[-1,0,0]),('leftUpperLeg','leftLowerLeg',[0,-1,0]),('leftLowerLeg','leftFoot',[0,-1,0]),('rightUpperLeg','rightLowerLeg',[0,-1,0]),('rightLowerLeg','rightFoot',[0,-1,0]),('leftIndexProximal','leftIndexIntermediate',[1,0,0])]:
        if a in hb and c in hb:
            d=Wn[hb[c]][:3,3]-Wn[hb[a]][:3,3]; ang=math.degrees(math.acos(np.clip(d@t/np.linalg.norm(d),-1,1)))
            if ang>worst: worst,worst_name=ang,a
    # independent: old mesh skinned with the new skeleton (old inverse binds) == new mesh at its rest
    Wo_new=[None]*len(go['nodes'])
    vo=verts(go,bo,Wn); vn=verts(gn,bn,Wn)
    d=np.linalg.norm(vo-vn,axis=1)
    print(f"{new.split('/')[-1]}: worst limb off its T-pose direction {worst:.2f} deg ({worst_name}); new rest mesh vs old mesh posed by the new skeleton: max {d.max():.2e} m")
