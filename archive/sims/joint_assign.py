# Why per-track NORAD matching is non-unique, and how a joint fit with shared nuisances resolves it.
# Known receiver position. K simultaneous tracks of T seconds. Unknown common LO offset b.
# Per-satellite TLE along-track error -> time shift. Optional Starlink CFO corrections (E09-like).
import os, numpy as np
from scipy.optimize import linear_sum_assignment
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assoc.py')).read().split('res={}')[0])
rng=np.random.default_rng(21)
K,T,fs_hz,sig_n=4,3.0,10,5.0
dgrid=np.arange(-1.2,1.21,0.1)                       # TLE epoch-shift hypotheses [s]
def run(sig_s,cfo,trials=80):
    out={'pt_off':[], 'pt_off_amb':[], 'pt_shift':[], 'pt_shift_amb':[], 'joint':[], 'joint_amb':[]}
    for _ in range(trials):
        t0=rng.uniform(0,86400); el=elev(t0,lat,lon); cand=np.where(el>10)[0]
        hi=np.where(el>40)[0]
        if hi.size<K: continue
        truth=rng.choice(hi,K,replace=False)
        ts=np.arange(0,T+1e-9,1/fs_hz); tg=np.arange(-1.2,T+1.2+1e-9,0.1)
        d=rng.normal(0,sig_s,N)/7.6e3                    # TLE along-track error as time shift per satellite
        Ftrue=dop(t0+ts,lat,lon,np.zeros(N))[:,truth]    # [len(ts),K]
        Fp=dop(t0+tg,lat,lon,d)[:,cand]                  # predicted on padded grid [len(tg),C]
        b=rng.uniform(-30e3,30e3)
        meas=np.zeros((len(ts),K))
        for k in range(K):
            c=np.zeros(len(ts))
            if cfo:
                c+=np.repeat(rng.normal(0,30,int(np.ceil(T))+1),fs_hz)[:len(ts)]                 # zero-order 1-s jumps
                if rng.random()<0.3: c+=275*(2*((ts+rng.random())%1.0)-1)                         # sawtooth, 1-s period
            meas[:,k]=Ftrue[:,k]+b+c+rng.normal(0,sig_n,len(ts))
        # predicted at shifted times: index into padded grid (0.1 s grid, ts on 0.1 s grid)
        i0=np.round((ts+1.2)/0.1).astype(int)
        P=np.stack([Fp[i0+int(round(dd/0.1)),:] for dd in dgrid])      # [D, len(ts), C]
        ci={s:i for i,s in enumerate(cand)}; tidx=np.array([ci[s] for s in truth])
        R=meas[None,:,:,None]-P[:,:,None,:]                              # [D, t, K, C]
        mean=R.mean(1); var=R.var(1)                                     # [D,K,C]
        # (1) per-track, offset only, no epoch shift
        J=np.sqrt(var[len(dgrid)//2]); o=np.sort(J,1); a=J.argmin(1)
        out['pt_off'].append(np.mean(a==tidx)); out['pt_off_amb'].append(np.mean(o[:,1]/o[:,0]<2))
        # (2) per-track, offset + free per-satellite epoch shift (common practice)
        J2=np.sqrt(var.min(0)); o=np.sort(J2,1); a=J2.argmin(1)
        out['pt_shift'].append(np.mean(a==tidx)); out['pt_shift_amb'].append(np.mean(o[:,1]/o[:,0]<2))
        # (3) joint: one common b for all tracks, epoch shift with prior, one-to-one assignment
        pen=(sig_n**2)*(dgrid/(sig_s/7.6e3+1e-3))[:,None,None]**2
        bh=[]
        for k in range(K):
            top=np.argsort(J2[k])[:5]
            bh+=list(mean[:,k,top].ravel())
        best=[]
        for bb in bh:
            C=(var+(mean-bb)**2+pen).min(0)                              # [K,C]
            r,cidx=linear_sum_assignment(C); best.append((C[r,cidx].sum(),tuple(cidx)))
        best.sort(); tot,asg=best[0]
        # ambiguity per track at the chosen b: RMS-cost ratio of runner-up vs assigned candidate (<2 = ambiguous)
        bb=[x for x in bh if abs((var+(mean-x)**2+pen).min(0)[np.arange(K),list(asg)].sum()-tot)<1e-6*max(tot,1)][0]
        C=(var+(mean-bb)**2+pen).min(0); amb=[]
        for k in range(K):
            ck=np.sort(C[k]); amb.append(np.sqrt(ck[1]/max(C[k,asg[k]],1e-9))<2 if ck[0]==C[k,asg[k]] else True)
        out['joint'].append(np.mean(np.array(asg)==tidx)); out['joint_amb'].append(np.mean(amb))
    return {k:np.mean(v) for k,v in out.items()}
for sig_s,name in ((400,'SupGP-like 0.4 km'),(3000,'GP-like 3 km')):
    for cfo in (False,True):
        r=run(sig_s,cfo)
        print(f"{name:<17} CFO {'on ' if cfo else 'off'} | per-track offset-only: correct {r['pt_off']:.2f}, ambiguous {r['pt_off_amb']:.2f} | "
              f"per-track + free epoch shift: correct {r['pt_shift']:.2f}, ambiguous {r['pt_shift_amb']:.2f} | joint common-LO: correct {r['joint']:.2f}, ambiguous {r['joint_amb']:.2f}")
