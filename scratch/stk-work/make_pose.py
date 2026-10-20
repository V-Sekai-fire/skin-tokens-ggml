# Writes a copy of a VRM 1.0 with its limbs turned into an arbitrary pose (node rotations only).
import sys,json,struct
import numpy as np
from math import sin,cos,radians
src,dst=sys.argv[1],sys.argv[2]
b=open(src,'rb').read(); n=struct.unpack('<I',b[12:16])[0]; g=json.loads(b[20:20+n]); rest=b[20+n:]
hb={k:v['node'] for k,v in g['extensions']['VRMC_vrm']['humanoid']['humanBones'].items()}
def qmul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return [aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz]
def axis(ax,deg):
    a=radians(deg)/2; v=np.array(ax,float); v/=np.linalg.norm(v); return [*(v*sin(a)),cos(a)]
pose={'leftUpperArm':([0.3,0.2,1],-65),'rightUpperArm':([0.1,-0.4,1],70),'leftLowerArm':([0,1,0.5],50),'rightLowerArm':([0.2,1,0],-45),
      'leftUpperLeg':([0,0.3,1],25),'rightUpperLeg':([1,0,0.4],-35),'leftLowerLeg':([1,0,0],40),'rightLowerLeg':([1,0.2,0],30),
      'leftIndexProximal':([0,0,1],-35),'rightMiddleIntermediate':([0,0,1],40),'leftHand':([1,0,0],30)}
for k,(ax,deg) in pose.items():
    if k in hb:
        nd=g['nodes'][hb[k]]; nd['rotation']=qmul(nd.get('rotation',[0,0,0,1]),axis(ax,deg))
t=json.dumps(g).encode(); t+=b' '*(-len(t)%4)
open(dst,'wb').write(struct.pack('<III',0x46546C67,2,12+8+len(t)+len(rest))+struct.pack('<II',len(t),0x4E4F534A)+t+rest)
