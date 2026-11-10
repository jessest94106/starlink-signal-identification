# Single-track satellite ID success vs window length and prior errors
import os, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assoc.py')).read().split('res={}')[0])
rng=np.random.default_rng(11); res={}
for trial in range(100):
    t0=rng.uniform(0,86400); el=elev(t0,lat,lon); cand=np.where(el>10)[0]; j=rng.choice(np.where(el>25)[0])
    for Tw in (10,60,300):
        ts=t0+np.linspace(0,Tw,41)
        ft=dop(ts,lat,lon,np.zeros(N))[:,j]+20e3+rng.normal(0,1,ts.size)   # unknown LO offset, 1 Hz noise
        for pe,ae in ((1e3,1e3),(10e3,5e3),(50e3,5e3)):
            th=rng.uniform(0,2*np.pi); la=lat+pe*np.cos(th)/Re; lo=lon+pe*np.sin(th)/(Re*np.cos(lat))
            fp=dop(ts,la,lo,rng.normal(0,ae,N)/7.6e3)[:,cand]
            r=ft[:,None]-fp; A=np.stack([np.ones_like(ts),ts-ts.mean()],1)
            r=r-A@np.linalg.lstsq(A,r,rcond=None)[0]; J=np.sqrt((r**2).mean(0)); k=cand[np.argsort(J)]
            res.setdefault((Tw,pe/1e3,ae/1e3),[]).append((k[0]==j, j in k[:3]))
for key in sorted(res):
    v=np.array(res[key]); print(f"window {key[0]:>3} s, position err {key[1]:>4.0f} km, along-track err {key[2]:.0f} km: top-1 {v[:,0].mean():.2f}, top-3 {v[:,1].mean():.2f}")
