import numpy as np
rng=np.random.default_rng(1)
mu=3.986004418e14; Re=6378137.0; we=7.2921159e-5; c=299792458.0; fc=11.7e9; lam=c/fc
# Walker-like approximation of ~10k Starlink sats (incl,alt_km,planes,per_plane)
shells=[(53.0,550,72,22),(53.2,540,72,22),(70.0,570,36,20),(97.6,560,10,52),(43.0,530,48,50),(53.0,525,48,50),(33.0,525,28,40)]
els=[]
for inc,alt,P,S in shells:
    a=Re+alt*1e3; n=np.sqrt(mu/a**3)
    raan=np.repeat(np.arange(P)*2*np.pi/P,S); u0=(np.tile(np.arange(S),P)*2*np.pi/S+rng.uniform(0,2*np.pi,P).repeat(S))
    els.append((np.full(P*S,a),np.full(P*S,np.radians(inc)),raan,u0,np.full(P*S,n)))
a,inc,raan,u0,n=[np.concatenate(x) for x in zip(*els)]
N=a.size
def sat_eci(t):
    u=u0+n*t
    x=np.stack([np.cos(u),np.sin(u),np.zeros_like(u)],-1)
    # rotate by inc about x, then raan about z
    ci,si,cO,sO=np.cos(inc),np.sin(inc),np.cos(raan),np.sin(raan)
    X=x[:,0]; Y=x[:,1]*ci; Z=x[:,1]*si
    return a[:,None]*np.stack([X*cO-Y*sO, X*sO+Y*cO, Z],-1)
lat,lon=np.radians(47.65),np.radians(-122.31)
def rx_eci(t):
    th=lon+we*t
    return Re*np.array([np.cos(lat)*np.cos(th),np.cos(lat)*np.sin(th),np.sin(lat)])
def geom(t,dt=0.05):
    r=lambda tt: np.linalg.norm(sat_eci(tt)-rx_eci(tt),axis=1)
    rho=r(t); rdot=(r(t+dt)-r(t-dt))/(2*dt); rdd=(r(t+dt)-2*rho+r(t-dt))/dt**2
    up=rx_eci(t)/Re; los=(sat_eci(t)-rx_eci(t))/rho[:,None]; el=np.degrees(np.arcsin(los@up))
    return el,-rdot/lam,-rdd/lam
rows=[]
for t in rng.uniform(0,86400,300):
    el,f,fd=geom(t)
    for mask in (10,25):
        v=el>mask; fv=np.sort(f[v])
        gaps=np.diff(fv)
        # nearest-neighbour Doppler gap per visible sat
        nn=np.minimum(np.r_[np.inf,gaps],np.r_[gaps,np.inf])
        # tone comb: Doppler mod 43.945 kHz
        fm=np.sort(np.mod(f[v],43945.)); gm=np.diff(np.r_[fm,fm[0]+43945.])
        nnm=np.minimum(gm,np.r_[gm[-1],gm[:-1]])
        rows.append((mask,v.sum(),np.median(nn),np.mean(nn<1000),np.median(nnm),np.abs(f[v]).max(),np.abs(fd[v]).max(),np.median(np.abs(fd[v]))))
R=np.array(rows)
for mask in (10,25):
    m=R[:,0]==mask
    print(f"mask {mask} deg: visible median {np.median(R[m,1]):.0f} (5-95%: {np.percentile(R[m,1],5):.0f}-{np.percentile(R[m,1],95):.0f}); "
          f"median NN Doppler gap {np.median(R[m,2])/1e3:.1f} kHz; frac sats with a neighbour <1 kHz {np.mean(R[m,3]):.2f}; "
          f"median NN gap of Doppler mod 43.945k {np.median(R[m,4]):.0f} Hz; max|fD| {np.max(R[m,5])/1e3:.0f} kHz; max|fDdot| {np.max(R[m,6]):.0f} Hz/s; median|fDdot| {np.median(R[m,7]):.0f} Hz/s")
print("N sats simulated",N)
# sensitivity: 10 km receiver error and 5 km along-track error, at el>25
t=12345.; el,f,fd=geom(t); v=el>25
lat0=lat; 
def dop_at(latr,lonr,shift=0.0):
    global lat,lon
    L0,O0=lat,lon; lat,lon=latr,lonr; e,ff,_=geom(t+shift); lat,lon=L0,O0; return ff
f_pos=dop_at(lat+10e3/Re,lon)          # 10 km north error
print("10 km north position error -> |dfD| over visible sats: median %.0f Hz, max %.0f Hz"%(np.median(np.abs(f_pos-f)[v]),np.abs(f_pos-f)[v].max()))
print("5 km along-track (0.66 s) error -> |dfD| median %.0f Hz, max %.0f Hz"%(np.median(np.abs(fd[v])*5e3/7.6e3),np.max(np.abs(fd[v]))*5e3/7.6e3))
