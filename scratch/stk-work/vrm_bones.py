import glob,math,sys
import numpy as np
exec(open('frame_check.py').read().split('for p in sys.argv')[0])
for p in sys.argv[1:]:
    g,b=load(p); N=g['nodes']; hb={k:v['node'] for k,v in g['extensions']['VRMC_vrm']['humanoid']['humanBones'].items()}
    par={c:i for i,n in enumerate(N) for c in n.get('children',[])}
    def W(i):
        n=N[i]; M=np.eye(4); M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0]); return W(par[i])@M if i in par else M
    bad=[]
    for a,c in [('hips','spine'),('spine','chest'),('leftUpperArm','leftLowerArm'),('leftLowerArm','leftHand'),('rightUpperArm','rightLowerArm'),('leftUpperLeg','leftLowerLeg'),('leftLowerLeg','leftFoot'),('rightUpperLeg','rightLowerLeg'),('neck','head')]:
        if a in hb and c in hb:
            Wa=W(hb[a]); d=W(hb[c])[:3,3]-Wa[:3,3]; y=Wa[:3,1]
            ang=math.degrees(math.acos(np.clip(y@d/np.linalg.norm(d)/np.linalg.norm(y),-1,1)))
            if ang>5: bad.append(f"{a}->{c} {ang:.0f}deg")
    print(p.split('/')[-1], bad)
