import json,struct,sys
import numpy as np
def load(p):
    b=open(p,'rb').read(); l=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+l])
    off=20+l; bl=struct.unpack('<I',b[off:off+4])[0]; bin_=b[off+8:off+8+bl]
    def acc(i):
        a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]
        n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
        dt={5126:np.float32,5123:np.uint16,5125:np.uint32,5121:np.uint8}[a['componentType']]
        st=bv.get('byteOffset',0)+a.get('byteOffset',0)
        return np.frombuffer(bin_,dtype=dt,count=a['count']*n,offset=st).reshape(a['count'],n) if n>1 else np.frombuffer(bin_,dtype=dt,count=a['count'],offset=st)
    pr=j['meshes'][0]['primitives'][0]['attributes']; idx=j['meshes'][0]['primitives'][0]['indices']
    return acc(pr['POSITION']),acc(pr['JOINTS_0']).astype(int),acc(pr['WEIGHTS_0']),acc(idx).reshape(-1,3)
def dense(J,W,nj):
    D=np.zeros((len(J),nj)); 
    for s in range(4): np.add.at(D,(np.arange(len(J)),J[:,s]),W[:,s])
    return D
