# Pilot-tone version: each track is one tone of a 9-tone comb, unknown index k in -4..4 (dF = 43.945 kHz).
# Known receiver position, unknown common LO offset b, per-satellite TLE along-track error. No CFO (tones are not CFO-corrected, A03).
# Compares: per-track offset-only; joint common b with residual wrapped mod dF (k unknown); joint common b with k known (comb centre found).
import os, numpy as np
from scipy.optimize import linear_sum_assignment
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assoc.py')).read().split('res={}')[0])
rng=np.random.default_rng(33); dF=43945.0; K=4; fs_hz=10; sig_n=5.0
dgrid=np.arange(-1.2,1.21,0.1)
wrap=lambda x: (x+dF/2)%dF-dF/2
def run(sig_s,T,trials=60):
    o={k:[] for k in ('pt','pt_amb','jw','jw_amb','jk','jk_amb')}
    for _ in range(trials):
        t0=rng.uniform(0,86400); el=elev(t0,lat,lon); cand=np.where(el>10)[0]; hi=np.where(el>40)[0]
        if hi.size<K: continue
        truth=rng.choice(hi,K,replace=False); ts=np.arange(0,T+1e-9,1/fs_hz); tg=np.arange(-1.2,T+1.2+1e-9,0.1)
        d=rng.normal(0,sig_s,N)/7.6e3
        Ft=dop(t0+ts,lat,lon,np.zeros(N))[:,truth]; Fp=dop(t0+tg,lat,lon,d)[:,cand]
        b=rng.uniform(-20e3,20e3); kk=rng.integers(-4,5,K)
        meas=Ft+b+kk*dF+rng.normal(0,sig_n,Ft.shape)
        i0=np.round((ts+1.2)/0.1).astype(int); P=np.stack([Fp[i0+int(round(x/0.1)),:] for x in dgrid])
        ci={s:i for i,s in enumerate(cand)}; tidx=np.array([ci[s] for s in truth])
        R=meas[None,:,:,None]-P[:,:,None,:]; mean=R.mean(1); var=R.var(1)
        pen=(sig_n**2)*(dgrid/(sig_s/7.6e3+1e-3))[:,None,None]**2
        J=np.sqrt(var[len(dgrid)//2]); s_=np.sort(J,1)
        o['pt'].append(np.mean(J.argmin(1)==tidx)); o['pt_amb'].append(np.mean(s_[:,1]/s_[:,0]<2))
        Jsh=np.sqrt((var+pen).min(0))
        for key,mfun in (('jw',lambda m,bb: wrap(m-bb)),('jk',lambda m,bb: m-bb)):
            mm=mean if key=='jw' else mean-(kk*dF)[None,:,None]            # 'jk': comb index known
            bh=[]
            for k in range(K): bh+=list(mm[:,k,np.argsort(Jsh[k])[:5]].ravel())
            best=None
            for bb in bh:
                C=(var+mfun(mm,bb)**2+pen).min(0); r,c=linear_sum_assignment(C); tot=C[r,c].sum()
                if best is None or tot<best[0]: best=(tot,c,C)
            tot,asg,C=best; amb=[]
            for k in range(K):
                ck=np.sort(C[k]); amb.append(not (ck[0]==C[k,asg[k]] and np.sqrt(ck[1]/max(ck[0],1e-9))>=2))
            o[key].append(np.mean(asg==tidx)); o[key+'_amb'].append(np.mean(amb))
    return {k:np.mean(v) for k,v in o.items()}
for sig_s,name in ((400,'SupGP-like 0.4 km'),(3000,'GP-like 3 km')):
    for T in (3.0,10.0):
        r=run(sig_s,T)
        print(f"{name:<17} T={T:>4.0f} s | per-track: {r['pt']:.2f} correct, {r['pt_amb']:.2f} ambiguous | joint, k unknown (wrapped): {r['jw']:.2f}, {r['jw_amb']:.2f} | joint, k known: {r['jk']:.2f}, {r['jk_amb']:.2f}")
