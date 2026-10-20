import sys,json,struct,math
import numpy as np
exec(open('frame_check.py').read().split('for p in sys.argv')[0])
def check(p):
    g,b=load(p); N=g['nodes']; ext=g.get('extensions',{})
    if 'VRMC_vrm' in ext: ver='1.0'; hb={k:v['node'] for k,v in ext['VRMC_vrm']['humanoid']['humanBones'].items()}
    elif 'VRM' in ext: ver='0.x'; hb={h['bone']:h['node'] for h in ext['VRM']['humanoid']['humanBones']}
    else: ver='none'; hb={}
    par={c:i for i,n in enumerate(N) for c in n.get('children',[])}
    def W(i):
        n=N[i]; M=np.eye(4)
        if 'matrix' in n: M=np.array(n['matrix']).reshape(4,4).T
        else: M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0])
        return W(par[i])@M if i in par else M
    P=lambda k: W(hb[k])[:3,3]
    left=P('leftUpperArm')-P('rightUpperArm'); up=P('head')-P('hips')
    fwd=np.cross(left,up); fwd/=np.linalg.norm(fwd)
    toe=None
    if 'leftToes' in hb: t=P('leftToes')-P('leftFoot'); toe='+Z' if t[2]>0 else '-Z'
    # winding
    vol=0
    for i,n in enumerate(N):
        if 'mesh' not in n: continue
        M=np.eye(4) if 'skin' in n else W(i)
        for pr in g['meshes'][n['mesh']]['primitives']:
            if pr.get('mode',4)!=4 or 'indices' not in pr: continue
            v=acc(g,b,pr['attributes']['POSITION']); v=(M[:3,:3]@v.T).T+M[:3,3]
            idx=acc(g,b,pr['indices']).astype(int).reshape(-1,3); a,bb,c=v[idx[:,0]],v[idx[:,1]],v[idx[:,2]]
            vol+=np.einsum('ij,ij->i',a,np.cross(bb,c)).sum()/6
    # joint frames along bone over humanoid chains
    chains=[('hips','spine'),('spine','chest'),('leftUpperArm','leftLowerArm'),('leftLowerArm','leftHand'),('rightUpperArm','rightLowerArm'),('leftUpperLeg','leftLowerLeg'),('leftLowerLeg','leftFoot'),('rightUpperLeg','rightLowerLeg'),('neck','head')]
    ok=0;tot=0
    for a,c in chains:
        if a in hb and c in hb:
            Wa=W(hb[a]); d=P(c)-Wa[:3,3]; y=Wa[:3,1]/np.linalg.norm(Wa[:3,1]); tot+=1; ok+= (y@d/np.linalg.norm(d))>math.cos(math.radians(5))
    rot=sum(1 for k,i in hb.items() if abs(N[i].get('rotation',[0,0,0,1])[3])<0.99999)
    print(f"{p.split('/')[-1]}: VRM {ver}; faces {'+Z' if fwd[2]>0.5 else '-Z' if fwd[2]<-0.5 else np.round(fwd,2)} (toes {toe}); left on {'+X' if left[0]>0 else '-X'}; up +Y {up[1]>0}; winding {'CCW' if vol>0 else 'CW'}; bones with +Y along bone {ok}/{tot}; humanoid bones with a rest turn {rot}/{len(hb)}")
for p in sys.argv[1:]:
    try: check(p)
    except Exception as e: print(p.split('/')[-1], 'ERROR', e)
