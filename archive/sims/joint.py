import numpy as np
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assoc.py')).read().split('res={}')[0])
rng=np.random.default_rng(5)
def dop_sub(ts,la,lo,d,idx):
    out=[]
    for t in ts:
        def r(tt):
            u=u0[idx]+n[idx]*(tt+d[idx]); ci,si,cO,sO=np.cos(inc[idx]),np.sin(inc[idx]),np.cos(raan[idx]),np.sin(raan[idx])
            X=np.cos(u);Y=np.sin(u)*ci;Z=np.sin(u)*si
            s=a[idx,None]*np.stack([X*cO-Y*sO,X*sO+Y*cO,Z],-1); return np.linalg.norm(s-rxe(tt,la,lo),axis=1)
        out.append(-(r(t+0.05)-r(t-0.05))/0.1/lam)
    return np.array(out)
K=3; Tw=60; out=[]
for trial in range(25):
    t0=rng.uniform(0,86400); el=elev(t0,lat,lon); cand=np.where(el>10)[0]
    js=rng.choice(np.where(el>25)[0],K,replace=False); ts=t0+np.linspace(0,Tw,31)
    ft=dop_sub(ts,lat,lon,np.zeros(N),js)+20e3+rng.normal(0,1,(ts.size,K))      # common LO bias
    th=rng.uniform(0,2*np.pi); la0=lat+50e3*np.cos(th)/Re; lo0=lon+50e3*np.sin(th)/(Re*np.cos(lat))   # 50 km prior
    d=rng.normal(0,5e3,N)/7.6e3
    A=np.stack([np.ones_like(ts),ts-ts.mean()],1); P=np.eye(ts.size)-A@np.linalg.pinv(A)
    best=(np.inf,None); single=None
    for dn in np.arange(-80e3,80.1e3,8e3):
        for de in np.arange(-80e3,80.1e3,8e3):
            la=la0+dn/Re; lo=lo0+de/(Re*np.cos(lat))
            fp=dop_sub(ts,la,lo,d,cand)
            J=np.stack([np.sqrt(((P@(ft[:,k:k+1]-fp))**2).mean(0)) for k in range(K)])   # K x Ncand
            asg=J.argmin(1); cost=J.min(1).sum()
            if len(set(asg))<K: cost=np.inf
            if dn==0 and de==0: single=cand[asg]
            if cost<best[0]: best=(cost,cand[asg],dn,de)
    out.append((np.all(single==js), np.all(best[1]==js)))
o=np.array(out); print(f"K={K} tracks, 60 s, 50 km prior, 5 km along-track: per-track ID at prior {o[:,0].mean():.2f}; joint position-grid ID {o[:,1].mean():.2f} (n={len(o)})")
