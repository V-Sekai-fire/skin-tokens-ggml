import json,struct,sys
import numpy as np
def rd(p):
    b=open(p,'rb').read(); n=struct.unpack('<I',b[12:16])[0]; return json.loads(b[20:20+n])
def q2m(q):
    x,y,z,w=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
for p in sys.argv[1:]:
    g=rd(p); N=g['nodes']; J=g['skins'][0]['joints'] if g.get('skins') else list(range(len(N)))
    par={c:i for i,n in enumerate(N) for c in n.get('children',[])}
    def W(i):
        n=N[i]; M=np.eye(4); M[:3,:3]=q2m(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])); M[:3,3]=n.get('translation',[0,0,0])
        return W(par[i])@M if i in par else M
    P={j:W(j)[:3,3] for j in J}
    ys=np.array([P[j][1] for j in J]); lo=ys.min(); h=ys.max()-lo
    feet=[j for j in J if not [c for c in N[j].get('children',[]) if c in J] and P[j][1]<lo+0.08*h]
    fw=[P[j]-P[par[j]] for j in feet if j in par]
    f=np.mean([v*[1,0,1] for v in fw],0) if fw else np.zeros(3)
    named=[(N[j].get('name'),round(float(P[j][0]),3)) for j in J if N[j].get('name','').startswith(('Left','Right'))][:2]
    print(f"{p.split('/')[-1]}: foot-tip joints {len(feet)}, toe direction (x,z) = ({f[0]:+.3f}, {f[2]:+.3f}) -> faces {'+Z' if f[2]>0 else '-Z'}; {named}")
