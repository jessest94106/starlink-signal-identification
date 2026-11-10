import numpy as np
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'overlap.py')).read().split('rows=[]')[0])
rng=np.random.default_rng(7)
def sat_eci_off(t,d):
    u=u0+n*(t+d); ci,si,cO,sO=np.cos(inc),np.sin(inc),np.cos(raan),np.sin(raan)
    X=np.cos(u); Y=np.sin(u)*ci; Z=np.sin(u)*si
    return a[:,None]*np.stack([X*cO-Y*sO, X*sO+Y*cO, Z],-1)
def rxe(t,la,lo):
    th=lo+we*t; return Re*np.array([np.cos(la)*np.cos(th),np.cos(la)*np.sin(th),np.sin(la)])
def dop(ts,la,lo,d):
    out=[]
    for t in ts:
        r=lambda tt: np.linalg.norm(sat_eci_off(tt,d)-rxe(tt,la,lo),axis=1)
        out.append(-(r(t+0.05)-r(t-0.05))/0.1/lam)
    return np.array(out)       # [T, N]
def elev(t,la,lo):
    p=rxe(t,la,lo); s=sat_eci_off(t,0*n); los=s-p; los/=np.linalg.norm(los,axis=1)[:,None]
    return np.degrees(np.arcsin(los@(p/Re)))
res={}
for trial in range(60):
    t0=rng.uniform(0,86400); el=elev(t0,lat,lon); cand=np.where(el>10)[0]; trueset=np.where(el>25)[0]
    j=rng.choice(trueset)
    for Tw in (1,10,60):
        ts=t0+np.arange(0,Tw+1e-9,max(0.5,Tw/40))
        ft=dop(ts,lat,lon,np.zeros(N))[:,j]+20e3+rng.normal(0,1,ts.size)   # +20 kHz unknown LO bias, 1 Hz noise
        for pe,ae in ((0,0),(10e3,5e3),(50e3,5e3)):
            th=rng.uniform(0,2*np.pi); la=lat+pe*np.cos(th)/Re; lo=lon+pe*np.sin(th)/(Re*np.cos(lat))
            d=rng.normal(0,ae,N)/7.6e3
            fp=dop(ts,la,lo,d)[:,cand]
            for mode in ('bias known','bias fitted'):
                r=ft[:,None]-fp-(20e3 if mode=='bias known' else 0)
                if mode=='bias fitted':
                    A=np.stack([np.ones_like(ts),ts-ts.mean()],1) if Tw>1 else np.ones((ts.size,1))
                    r=r-A@np.linalg.lstsq(A,r,rcond=None)[0]
                J=np.sqrt((r**2).mean(0)); k=np.argsort(J)
                ok=cand[k[0]]==j
                res.setdefault((Tw,pe,ae,mode),[]).append(ok)
for key in sorted(res):
    print(key, f"correct ID {np.mean(res[key]):.2f}")
