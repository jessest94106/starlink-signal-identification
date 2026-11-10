# Satellite ID when the receiver position is KNOWN: effect of TLE quality, LO offset handling, beam gating, window length.
import os, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assoc.py')).read().split('res={}')[0])
rng=np.random.default_rng(3); res={}; ncand={}
def off_boresight(t):            # boresight = local zenith -> off-boresight angle = 90 - elevation
    return 90-elev(t,lat,lon)
for trial in range(150):
    t0=rng.uniform(0,86400); el=elev(t0,lat,lon)
    allc=np.where(el>10)[0]; gate=np.where(el>60)[0]            # wide-beam LNB at zenith, accept <30 deg off boresight
    if gate.size<2: continue
    j=rng.choice(gate)
    for Tw in (1,10):
        ts=t0+np.linspace(0,Tw,21)
        ft=dop(ts,lat,lon,np.zeros(N))[:,j]+7e3+rng.normal(0,1,ts.size)     # 7 kHz LO offset (true), 1 Hz noise
        for tle,sig in (('SupGP-like 0.4 km',0.4e3),('GP-like 3 km',3e3)):
            fp=dop(ts,lat,lon,rng.normal(0,sig,N)/7.6e3)
            for lo in ('LO unknown (fit offset)','LO known to 20 Hz'):
                for cs,cand in (('all >10 deg',allc),('beam-gated',gate)):
                    r=ft[:,None]-fp[:,cand]
                    if lo.startswith('LO unknown'):
                        r=r-r.mean(0)   # LO drift over <=10 s is negligible; keep Doppler-rate information
                    else:
                        r=r-(7e3+rng.normal(0,20))
                    J=np.sqrt((r**2).mean(0)); o=np.argsort(J)
                    res.setdefault((Tw,tle,lo,cs),[]).append((cand[o[0]]==j, J[o[1]]/max(J[o[0]],1e-9)))
                    ncand.setdefault(cs,[]).append(cand.size)
print({k:int(np.median(v)) for k,v in ncand.items()}, 'median candidates')
for k in sorted(res):
    v=np.array(res[k]); print(f"{k[0]:>2} s | {k[1]:<17} | {k[2]:<29} | {k[3]:<11} | correct {v[:,0].mean():.2f} | 2nd/1st cost ratio median {np.median(v[:,1]):.1f}")
