import json,struct,sys,math
import numpy as np
def rd(p):
    b=open(p,'rb').read(); n=struct.unpack('<I',b[12:16])[0]; return json.loads(b[20:20+n])
def q2m(q):
    x,y,z,w=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
for p in sys.argv[1:]:
    g=rd(p); J=g['skins'][0]['joints'] if g.get('skins') else list(range(len(g['nodes']))); N=g['nodes']
    par={c:i for i,n in enumerate(N) for c in n.get('children',[])}
    W={}
    def world(i):
        if i in W: return W[i]
        n=N[i]; M=np.eye(4)
        if 'matrix' in n: M=np.array(n['matrix']).reshape(4,4).T
        else:
            M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0])
        W[i]=(world(par[i])@M) if i in par else M; return W[i]
    rot=sum(1 for j in J if 'rotation' in N[j] and abs(N[j]['rotation'][3])<0.99999)
    along=0; tot=0; worst=[]
    for j in J:
        kids=[c for c in N[j].get('children',[]) if c in J]
        if len(kids)!=1: continue
        Wj=world(j); d=world(kids[0])[:3,3]-Wj[:3,3]
        if np.linalg.norm(d)<1e-6: continue
        y=Wj[:3,1]/np.linalg.norm(Wj[:3,1]); ang=math.degrees(math.acos(np.clip(y@d/np.linalg.norm(d),-1,1)))
        tot+=1; along+= ang<5; worst.append((ang,N[j].get('name')))
    worst.sort(reverse=True)
    names=[N[j].get('name') for j in J]
    print(f"{p.split('/')[-1]}: joints={len(J)} rotated_rest={rot} single-child bones with +Y along bone (<5deg): {along}/{tot}; worst {[(round(a),n) for a,n in worst[:3]]}; names {names[:6]}")
