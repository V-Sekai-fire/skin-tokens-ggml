import sys; sys.path.insert(0,'/Users/ernest.lee/Desktop/stk-work')
from glbw import *
import numpy as np
a,b,nj=sys.argv[1],sys.argv[2],int(sys.argv[3])
P,J0,W0,F=load(a); _,J1,W1,_=load(b)
D0=dense(J0,W0,nj).sum(0); D1=dense(J1,W1,nj).sum(0)
lost=[(j,round(float(D0[j]),1),round(float(D1[j]),1)) for j in range(nj) if D0[j]>1 and D1[j]<0.5*D0[j]]
print('joints losing >50% of their vertex weight:',lost if lost else 'none')
