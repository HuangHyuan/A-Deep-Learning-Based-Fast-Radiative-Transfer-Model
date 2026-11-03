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

"""AutoDL LM-1DVar retrieval for one station: parallel ARMS + GPU emulator."""
import os, sys, time, argparse
import numpy as np
from pathlib import Path
import torch, joblib
WORK = Path(os.environ.get('QJRMS_RETRIEVAL_WORK', _QJRMS_ROOT / 'data' / 'retrieval_work'))
MODEL_DIR = Path(os.environ.get('QJRMS_MODEL_DIR', _QJRMS_ROOT / 'model_assets'))
ARMS = Path(os.environ.get('ARMS_ROOT', _QJRMS_ROOT / 'external' / 'ARMS-gb'))
sys.path.insert(0, str(_QJRMS_CORE))
import arms_batch as AB
from GB_model import UnifiedAllSkyRTM_Transformer
NLEV, NCHAN, NS = 37, 22, 74
CLO = np.r_[np.full(NLEV, 180.0), np.full(NLEV, 1e-8)]
CHI = np.r_[np.full(NLEV, 330.0), np.full(NLEV, 30.0)]
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

def load_tf():
    m = UnifiedAllSkyRTM_Transformer(hidden_dim=160, nhead=4, num_transformer_layers=3)
    ck = torch.load(MODEL_DIR/'unified_model_jacobian_finetuned.pth', map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model_state_dict']); m.eval().to(DEV)
    a = joblib.load(MODEL_DIR/'scaler_atm_global.pkl'); s = joblib.load(MODEL_DIR/'scaler_sfc_global.pkl')
    h = joblib.load(MODEL_DIR/'scaler_hydro_cloudy.pkl'); y = joblib.load(MODEL_DIR/'scaler_y.pkl')
    return m, dict(amn=np.array([x.data_min_ for x in a], np.float32), amx=np.array([x.data_max_ for x in a], np.float32),
                   smn=np.asarray(s.data_min_, np.float32), smx=np.asarray(s.data_max_, np.float32),
                   hmn=np.array([x.data_min_ for x in h], np.float32), hmx=np.array([x.data_max_ for x in h], np.float32),
                   ymn=np.asarray(y.data_min_, np.float32), ymx=np.asarray(y.data_max_, np.float32))

class TFEval:
    def __init__(self, st, idx, prep):
        F = np.load(WORK/('fields_%d.npz' % st))
        self.idx = idx
        self.p = torch.tensor(np.asarray(F['p'][idx], np.float32), device=DEV)
        z = F['z'][idx]; dz = np.abs(z[:, :-1]-z[:, 1:]); lc = lambda a: 0.5*(a[:, :-1]+a[:, 1:])
        hyd = np.stack([lc(F['clwc'][idx])*dz, lc(F['crwc'][idx])*dz, lc(F['cswc'][idx])*dz, lc(F['ciwc'][idx])*dz], -1)
        for X in np.split(hyd, 4, -1): X[X < 1e-8] = 0
        self.hyd = hyd
        self.tclw = prep['tclw'][idx]; self.tcrw = prep['tcrw'][idx]
        sfc = np.stack([prep['t2m'][idx], prep['q2m'][idx], prep['sp'][idx], self.tclw, self.tcrw], -1)
        self.m, self.sc = load_tf()
        ar = np.where(self.sc['amx']-self.sc['amn'] > 0, self.sc['amx']-self.sc['amn'], 1.0)
        sr = np.where(self.sc['smx']-self.sc['smn'] > 0, self.sc['smx']-self.sc['smn'], 1.0)
        hr = np.where(self.sc['hmx']-self.sc['hmn'] > 0, self.sc['hmx']-self.sc['hmn'], 1.0)
        self.t_amn = torch.tensor(self.sc['amn'], device=DEV); self.t_ar = torch.tensor(ar, device=DEV)
        self.s = torch.tensor(((sfc-self.sc['smn'])/sr).astype(np.float32), device=DEV)
        self.h = torch.tensor(((hyd-self.sc['hmn'][None])/hr[None]).astype(np.float32), device=DEV)
        self.clear = (np.maximum.reduce([F['clwc'][idx].max(1), F['ciwc'][idx].max(1),
                                         F['cswc'][idx].max(1), F['crwc'][idx].max(1)]) <= 1e-8)
        self.scT = torch.tensor((self.sc['ymx']-self.sc['ymn'])[:, None]/ar[None, :, 1], device=DEV)
        self.scQ = torch.tensor((self.sc['ymx']-self.sc['ymn'])[:, None]/ar[None, :, 0], device=DEV)
        self.ymn = torch.tensor(self.sc['ymn'], device=DEV); self.yr = torch.tensor(self.sc['ymx']-self.sc['ymn'], device=DEV)
        self.M = len(idx)
    def _atm(self, states, lo=0):
        T = torch.as_tensor(states[:, :NLEV], dtype=torch.float32, device=DEV)
        q = torch.as_tensor(states[:, NLEV:], dtype=torch.float32, device=DEV)
        atm = torch.stack([q, T, self.p[lo:lo+states.shape[0]]], -1)
        return (atm - self.t_amn[None])/self.t_ar[None]
    def __call__(self, states, idx=None, need_jac=False):
        M = states.shape[0]; Y = np.empty((M, NCHAN), np.float32)
        K = np.empty((M, NCHAN, NS), np.float32) if need_jac else None
        ci = np.where(self.clear)[0]; yi = np.where(~self.clear)[0]
        bs = 256
        with torch.no_grad():
            for lo in range(0, M, bs):
                hi = min(lo+bs, M); am = self._atm(states[lo:hi], lo)
                cc = ci[(ci >= lo) & (ci < hi)] - lo; yy = yi[(yi >= lo) & (yi < hi)] - lo
                if len(cc):
                    o = self.m(am[cc], self.s[cc]); Y[lo+cc] = (o*self.yr+self.ymn).cpu().numpy()
                if len(yy):
                    o = self.m(am[yy], self.s[yy], self.h[yy]); Y[lo+yy] = (o*self.yr+self.ymn).cpu().numpy()
        if need_jac:
            for lo in range(0, M, bs):
                hi = min(lo+bs, M); am = self._atm(states[lo:hi], lo)
                cc = ci[(ci >= lo) & (ci < hi)] - lo; yy = yi[(yi >= lo) & (yi < hi)] - lo
                if len(cc):
                    s_c = self.s[lo+cc]
                    fc = lambda a, sv: self.m(a.unsqueeze(0), sv.unsqueeze(0)).squeeze(0)
                    J = torch.func.vmap(torch.func.jacrev(fc, argnums=0), in_dims=(0, 0))(am[cc], s_c)
                    K[lo+cc, :, :NLEV] = (J[:, :, :, 1]*self.scT[None]).detach().cpu().numpy()
                    K[lo+cc, :, NLEV:] = (J[:, :, :, 0]*self.scQ[None]).detach().cpu().numpy()
                if len(yy):
                    s_c = self.s[lo+yy]; h_c = self.h[lo+yy]
                    fc = lambda a, sv, hv: self.m(a.unsqueeze(0), sv.unsqueeze(0), hv.unsqueeze(0)).squeeze(0)
                    J = torch.func.vmap(torch.func.jacrev(fc, argnums=0), in_dims=(0, 0, 0))(am[yy], s_c, h_c)
                    K[lo+yy, :, :NLEV] = (J[:, :, :, 1]*self.scT[None]).detach().cpu().numpy()
                    K[lo+yy, :, NLEV:] = (J[:, :, :, 0]*self.scQ[None]).detach().cpu().numpy()
        return Y, K, 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('station', type=int)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--pool', type=int, default=32)
    ap.add_argument('--models', default='ARMS,Transformer')
    ap.add_argument('--capT', type=float, default=10.0); ap.add_argument('--capQ', type=float, default=2.0)
    a = ap.parse_args(); st = a.station; t00 = time.time()
    prep = dict(np.load(WORK/('prep_%d.npz' % st), allow_pickle=True))
    keep = prep['keep']; idx = np.where(keep)[0]; M = len(idx)
    xb = prep['xb']; B = prep['B']; Binv = np.linalg.inv(B)
    Rprof = 0.5**2 + (prep['b_c'][None, :]**2 + (prep['a_c'][None, :]*np.minimum(prep['tclw'][idx], prep['clip'])[:, None])**2) + 1.0**2
    RinvP = (1.0/Rprof).astype(np.float64)
    obs = prep['obs'][idx]; era5 = prep['era5'][idx]; yb = prep['Yb'][idx]
    tr = prep['tr'][idx]; te = prep['te'][idx]
    CAP = np.r_[np.full(NLEV, a.capT), np.full(NLEV, a.capQ)]
    os.environ['ARMS_FIELDS'] = str(WORK/('fields_%d.npz' % st)); AB.NPZ = os.environ['ARMS_FIELDS']
    ev_arms = AB.ArmsEvaluator(a.pool) if 'ARMS' in a.models else None
    ev_tf = TFEval(st, idx, prep) if 'Transformer' in a.models else None
    print('[%d] M=%d  cap=%.0fK/%.1fg  pool=%d' % (st, M, a.capT, a.capQ, a.pool), flush=True)
    _c0 = 0.5*np.sum((obs-prep['Yb'][idx])**2*RinvP, 1)
    print('   DIAG: sqrt(R) median=%.4f  (clear median %.4f)  tclw clip=%.4g  b_c median=%.4f  a_c median=%.1f' % (
        np.median(np.sqrt(Rprof)), np.median(np.sqrt(Rprof[prep['clear'][idx]])), prep['clip'], np.median(prep['b_c']), np.median(prep['a_c'])), flush=True)
    print('   DIAG: cost at xb using ARMS fwd: median=%.4f mean=%.1f sum=%.4g' % (np.median(_c0), _c0.mean(), _c0.sum()), flush=True)
    def cost(X, Y):
        dx = X-xb[None]; return 0.5*np.einsum('ni,ij,nj->n', dx, Binv, dx) + 0.5*np.sum((obs-Y)**2*RinvP, 1)
    def lm(name, ev, need_pool):
        ck = WORK/('ckpt_%d_%s.npz' % (st, name))
        it0 = 0
        if os.environ.get('RESUME', '0') == '1' and ck.exists():
            _c = np.load(ck); X = _c['X']; lam = _c['lam']; it0 = int(_c['it'])+1
            print('   [resume] %s from it%d' % (name, it0), flush=True)
        else:
            X = np.repeat(xb[None], M, 0); lam = np.full(M, 1e-2)
        for it in range(it0, a.iters):
            Y, K, bad = ev(X, idx, True)
            g = np.einsum('ij,nj->ni', Binv, X-xb[None]) - np.einsum('nij,nj->ni', K.transpose(0, 2, 1)*RinvP[:, None, :], obs-Y)
            KtR = K.transpose(0, 2, 1)*RinvP[:, None, :]
            H0 = Binv[None] + np.einsum('nij,njk->nik', KtR, K) + 1e-6*np.eye(NS)[None]
            Jc = cost(X, Y); done = np.zeros(M, bool)
            if need_pool: pass
            for trial in range(6):
                if done.all(): break
                H = H0 + lam[:, None, None]*np.eye(NS)[None]
                dX = np.clip(-np.linalg.solve(H, g[..., None])[..., 0], -CAP, CAP)
                Xn = np.clip(X+dX, CLO, CHI)
                Yn, _, _ = ev(Xn, idx, False)
                Jn = cost(Xn, Yn); take = (~done) & (Jn < Jc)
                X[take] = Xn[take]; Y[take] = Yn[take]; Jc[take] = Jn[take]
                lam[take] = np.maximum(lam[take]/3, 1e-6); lam[~take] = lam[~take]*10
                done |= take
            cum = np.abs(X-xb[None])
            print('   %s it%d J(med)=%.3f max|dX|=%.3f | cum mean|dT|=%.3f max|dT|=%.3f mean|dq|=%.4f (%.0f s)' % (
                name, it+1, np.median(Jc), np.abs(dX).max(), cum[:, :NLEV].mean(), cum[:, :NLEV].max(), cum[:, NLEV:].mean(), time.time()-t00), flush=True)
            np.savez_compressed(ck, X=X, lam=lam, it=it)
        Y, K, _ = ev(X, idx, True)
        KtR = K.transpose(0, 2, 1)*RinvP[:, None, :]
        H = Binv[None] + np.einsum('nij,njk->nik', KtR, K) + 1e-6*np.eye(NS)[None]
        dfs = np.trace(np.linalg.solve(H, KtR@K), axis1=1, axis2=2)
        return X, Y, dfs
    for name, ev, npool in (('ARMS', ev_arms, True), ('Transformer', ev_tf, False)):
        if ev is None: continue
        print('=== %s LM-GN ===' % name, flush=True)
        X, Y, dfs = lm(name, ev, npool)
        np.savez_compressed(WORK/('result_%d_%s.npz' % (st, name)), xa=X, yh=Y, dfs=dfs, idx=idx, obs=obs, era5=era5, yb=yb, tr=tr, te=te, xb=xb)
        dx = np.abs(X-xb[None])
        oa = np.sqrt(np.mean((obs-Y)**2, 1))
        skT = np.sqrt(np.mean((X[:, :NLEV]-era5[:, :NLEV])**2, 1)); skq = np.sqrt(np.mean((X[:, NLEV:]-era5[:, NLEV:])**2, 1))
        bgT = np.sqrt(np.mean((xb[None, :NLEV]-era5[:, :NLEV])**2, 1)); bgq = np.sqrt(np.mean((xb[None, NLEV:]-era5[:, NLEV:])**2, 1))
        print('   DFS med %.2f | mean|dT| %.3f max|dT| %.3f mean|dq| %.4f | O-A med %.3f' % (
            np.median(dfs), dx[:, :NLEV].mean(), dx[:, :NLEV].max(), dx[:, NLEV:].mean(), np.median(oa)), flush=True)
        for tag, m in (('train', tr), ('test', te), ('all', np.ones(M, bool))):
            print('   %-5s skill T %.3f (bg %.3f)  q %.4f (bg %.4f)' % (
                tag, np.median(skT[m]), np.median(bgT[m]), np.median(skq[m]), np.median(bgq[m])), flush=True)
    if ev_arms: ev_arms.close()
    print('[%d] DONE_STATION total %.0f s' % (st, time.time()-t00), flush=True)

if __name__ == '__main__':
    main()
