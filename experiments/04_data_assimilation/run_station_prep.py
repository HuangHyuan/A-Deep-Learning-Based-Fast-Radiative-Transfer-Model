#!/usr/bin/env python3
from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""AutoDL all-sky 1D-Var for one station: parallel ARMS (pool) + GPU emulator.
   R = sigma_noise^2 + [sigma_fwd_clear^2 + (a_c*TCLW_clip)^2] + sigma_rep^2 ; B = empirical (T-q kept)
   LM-GN, checkpointed, per-model immediate save."""
import os, sys, time, json, argparse
import numpy as np
from pathlib import Path
from netCDF4 import Dataset
import torch, joblib
WORK = Path(os.environ.get('QJRMS_RETRIEVAL_WORK', _QJRMS_ROOT / 'data' / 'retrieval_work'))
ARMS = Path(os.environ.get('ARMS_ROOT', _QJRMS_ROOT / 'external' / 'ARMS-gb'))
sys.path.insert(0, str(_QJRMS_CORE))
import arms_batch as AB
NLEV, NCHAN, NS = 37, 22, 74
SIG_NOISE, SIG_REP = 0.5, 1.0
CLO = np.r_[np.full(NLEV, 180.0), np.full(NLEV, 1e-8)]
CHI = np.r_[np.full(NLEV, 330.0), np.full(NLEV, 30.0)]

def load_station(st):
    p = WORK/'Dataset'/'REAL'/('%d_obs_era5_202501_202506.nc' % st)
    with Dataset(p) as ds:
        v = ds.variables
        pick = lambda *n: next(x for x in n if x in v)
        G = lambda *n: np.asarray(v[pick(*n)][:], float).T
        d = dict(p=G('Level_Pressure'), T=G('Level_Temperature'), q=G('Level_H2O'), o3=G('Level_O3'),
                 z=G('Geopotential_height', 'geopotential_height'),
                 clwc=np.maximum(G('Level_CLWC'), 0), ciwc=np.maximum(G('Level_CIWC'), 0),
                 cswc=np.maximum(G('Level_CSWC'), 0), crwc=np.maximum(G('Level_CRWC'), 0),
                 sp=np.asarray(v[pick('Surface_Pressure', 'surface_pressure')][:], float),
                 t2m=np.asarray(v['Temperature_2M'][:], float),
                 q2m=np.asarray(v[pick('H2O_2M', 'Obs_H2O_2M')][:], float),
                 freq=np.asarray(v['Central_Frequency'][:], float),
                 tclw=np.asarray(v['Level_CLWC'][:], float).sum(0),
                 tcrw=np.asarray(v['Level_CRWC'][:], float).sum(0))
        obs = np.asarray(v['Obs_BT'][:], float).T
        if 'Profile_Index' in v:
            pi = np.asarray(v['Profile_Index'][:], int); _, f = np.unique(pi, return_index=True); obs = obs[f]
        d['obs'] = obs
    N = d['p'].shape[0]
    out = {k: np.full((N, NLEV), np.nan) for k in ('T', 'q', 'o3', 'z', 'clwc', 'ciwc', 'cswc', 'crwc', 'p')}
    for i in range(N):
        val = np.where((d['p'][i] >= 1.0) & (d['p'][i] <= d['sp'][i] + 1e-3))[0]
        if val.size == 0: continue
        M = val.size
        idx = val[np.round(np.linspace(0, M-1, NLEV)).astype(int)] if M >= NLEV else np.r_[val, np.full(NLEV-M, val[-1])]
        for k in out: out[k][i] = d[k][i][idx]
    for k in out: d[k] = out[k]
    return d

def build_B(era5, tr):
    A = era5[tr] - era5[tr].mean(0)[None]
    D = A.std(0); D = np.where(D > 0, D, 1e-6); Z = A/D[None]
    C = np.corrcoef(Z, rowvar=False); C = 0.5*(C+C.T); C = 0.9*C + 0.1*np.eye(NS)
    w, V = np.linalg.eigh(C); w = np.maximum(w, 1e-6); C = (V*w)@V.T; np.fill_diagonal(C, 1.0)
    B = D[:, None]*C*D[None, :]; return 0.5*(B+B.T), D, C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('station', type=int)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--pool', type=int, default=32)
    ap.add_argument('--frac', type=float, default=0.70)
    ap.add_argument('--chi2', type=float, default=5.0)
    ap.add_argument('--no-tf', action='store_true')
    a = ap.parse_args()
    st = a.station; t00 = time.time()
    print('[%d] loading' % st, flush=True)
    d = load_station(st)
    era5 = np.concatenate([d['T'], d['q']], 1)
    obs = d['obs']; N = era5.shape[0]
    ok = np.isfinite(era5).all(1) & np.isfinite(obs).all(1) & (obs > 0).all(1) & (obs <= 330).all(1)
    keys = ('p', 'o3', 'z', 'clwc', 'ciwc', 'cswc', 'crwc')
    for k in keys + ('T', 'q'): d[k] = d[k][ok]
    d['sp'] = d['sp'][ok]; d['t2m'] = d['t2m'][ok]; d['q2m'] = d['q2m'][ok]
    d['tclw'] = d['tclw'][ok]; d['tcrw'] = d['tcrw'][ok]
    era5 = era5[ok]; obs = obs[ok]; N = era5.shape[0]
    np.savez(WORK/('fields_%d.npz' % st), p=d['p'], o3=d['o3'], z=d['z'], clwc=d['clwc'], ciwc=d['ciwc'],
             cswc=d['cswc'], crwc=d['crwc'], sp=d['sp'], t2m=d['t2m'], q2m=d['q2m'], freq=d['freq'])
    os.environ['ARMS_FIELDS'] = str(WORK/('fields_%d.npz' % st))
    AB.NPZ = os.environ['ARMS_FIELDS']
    ntr = int(round(a.frac*N)); tr = np.zeros(N, bool); tr[:ntr] = True; te = ~tr
    xb = np.concatenate([era5[tr][:, :NLEV].mean(0), era5[tr][:, NLEV:].mean(0)])
    B, D, C = build_B(era5, tr); Binv = np.linalg.inv(B)
    print('[%d] N=%d train=%d test=%d  B cond %.1e sigma_T %.2f-%.2f' % (st, N, ntr, N-ntr, np.linalg.cond(C), D[:NLEV].min(), D[:NLEV].max()), flush=True)
    ac = np.load(WORK/'ac_pooled.npz'); a_c = ac['a_c']; b_c = ac['b_c']; clip = float(ac['tclw_clip'])
    tclwc = np.minimum(d['tclw'], clip)
    sig_fwd = np.sqrt(b_c[None, :]**2 + (a_c[None, :]*tclwc[:, None])**2)
    Rprof = SIG_NOISE**2 + sig_fwd**2 + SIG_REP**2
    ev = AB.ArmsEvaluator(a.pool)
    t1 = time.time()
    Yb, Kb, bad = ev(np.repeat(xb[None], N, 0), np.arange(N), True)
    print('[%d] stage1 ARMS fwd+K on %d profiles in %.1f s (bad %d)' % (st, N, time.time()-t1, bad), flush=True)
    _Kd = np.einsum('nij,jk,nlk->nil', Kb, B, Kb)
    Sd = np.eye(NCHAN)[None]*Rprof[:, :, None] + _Kd
    w, V = np.linalg.eigh(Sd); w = np.maximum(w, 1e-12)
    Sdinv = (V/w[:, None, :])@V.transpose(0, 2, 1)
    chi2 = np.einsum('nc,ncd,nd->n', obs-Yb, Sdinv, obs-Yb)/NCHAN
    keep = chi2 <= a.chi2
    idx_all = np.where(keep)[0]
    print('[%d] QC chi2>%.1f rejects %d (%.2f%%) -> keep %d (train %d/test %d)' % (
        st, a.chi2, int((~keep).sum()), 100*(~keep).mean(), len(idx_all), int((keep&tr).sum()), int((keep&te).sum())), flush=True)
    np.savez_compressed(WORK/('prep_%d.npz' % st), xb=xb, B=B, a_c=a_c, b_c=b_c, clip=clip, tclw=d['tclw'],
                        era5=era5, obs=obs, tr=tr, te=te, clear=(np.maximum.reduce(
                            [d[k].max(1) for k in ('clwc','ciwc','cswc','crwc')]) <= 1e-8),
                        freq=d['freq'], sp=d['sp'], t2m=d['t2m'], q2m=d['q2m'], tcrw=d['tcrw'],
                        Yb=Yb, Kb=Kb, chi2=chi2, keep=keep)
    print('[%d] prep saved. total %.1f s' % (st, time.time()-t00), flush=True)
    ev.close()

if __name__ == '__main__':
    main()
