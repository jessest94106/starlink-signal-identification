# How many other satellites fall inside an antenna beam pointed at a target satellite (Seattle, el>25 deg targets)
import os, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assoc.py')).read().split('res={}')[0])
rng=np.random.default_rng(9); out={1.5:[],5:[],15:[],30:[]}
for t in rng.uniform(0,86400,200):
    p=rxe(t,lat,lon); s=sat_eci_off(t,0*n); los=s-p; los/=np.linalg.norm(los,axis=1)[:,None]
    el=np.degrees(np.arcsin(los@(p/Re))); vis=np.where(el>10)[0]; tgt=rng.choice(np.where(el>25)[0])
    ang=np.degrees(np.arccos(np.clip(los[vis]@los[tgt],-1,1)))
    for h in out: out[h].append(np.sum(ang<h)-1)
for h,v in out.items(): print(f"beam half-angle {h:>4} deg: other sats inside, mean {np.mean(v):.2f}, P(>=1) {np.mean(np.array(v)>=1):.2f}")
