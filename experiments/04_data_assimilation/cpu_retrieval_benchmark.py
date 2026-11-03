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

"""Legacy CPU-only benchmark using finite-difference BiLSTM Jacobians."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORK = Path(os.environ.get('QJRMS_RETRIEVAL_WORK', _QJRMS_ROOT / 'data' / 'retrieval_work'))
MODEL_DIR = Path(os.environ.get('QJRMS_MODEL_DIR', _QJRMS_ROOT / 'model_assets'))
sys.path.insert(0, str(_QJRMS_CORE))
from GB_model import UnifiedAllSkyRTM_Transformer, UnifiedAllSkyRTM_MLP, UnifiedAllSkyRTM_BiLSTM

NLEV, NCHAN, NS = 37, 22, 74
STATIONS = [50953, 52203, 59097, 58238, 54623]
SPECS = {
    'Transformer': (UnifiedAllSkyRTM_Transformer, dict(hidden_dim=160, nhead=4, num_transformer_layers=3), 'unified_model_jacobian_finetuned.pth'),
    'MLP': (UnifiedAllSkyRTM_MLP, dict(hidden_dim=12, num_mlp_blocks=3), 'mlp.pth'),
    'BiLSTM': (UnifiedAllSkyRTM_BiLSTM, dict(hidden_dim=180, num_bi_lstm_blocks=2), 'bilstm.pth'),
}
CLO = np.r_[np.full(NLEV, 180.0), np.full(NLEV, 1e-8)]
CHI = np.r_[np.full(NLEV, 330.0), np.full(NLEV, 30.0)]


def load_scalers():
    import joblib
    a = joblib.load(MODEL_DIR / 'scaler_atm_global.pkl')
    s = joblib.load(MODEL_DIR / 'scaler_sfc_global.pkl')
    h = joblib.load(MODEL_DIR / 'scaler_hydro_cloudy.pkl')
    y = joblib.load(MODEL_DIR / 'scaler_y.pkl')
    amn = np.array([x.data_min_ for x in a], np.float32)
    amx = np.array([x.data_max_ for x in a], np.float32)
    hmn = np.array([x.data_min_ for x in h], np.float32)
    hmx = np.array([x.data_max_ for x in h], np.float32)
    smn = np.asarray(s.data_min_, np.float32)
    smx = np.asarray(s.data_max_, np.float32)
    ymn = np.asarray(y.data_min_, np.float32)
    ymx = np.asarray(y.data_max_, np.float32)
    ar = np.where(amx - amn > 0, amx - amn, 1.0).astype(np.float32)
    sr = np.where(smx - smn > 0, smx - smn, 1.0).astype(np.float32)
    hr = np.where(hmx - hmn > 0, hmx - hmn, 1e-30).astype(np.float32)
    yr = (ymx - ymn).astype(np.float32)
    return dict(amn=amn, ar=ar, smn=smn, sr=sr, hmn=hmn, hr=hr, ymn=ymn, yr=yr,
                scT=yr[:, None] / ar[None, :, 1], scQ=yr[:, None] / ar[None, :, 0])


class Evaluator:
    def __init__(self, name, device, sc, st, idx, prep, batch):
        cls, kwargs, checkpoint = SPECS[name]
        self.name = name
        self.device = device
        self.sc = sc
        self.batch = batch
        self.prep = prep
        self.idx = idx
        self.clear = self._clear(st, idx)
        self.p = torch.tensor(np.asarray(np.load(WORK / f'fields_{st}.npz')['p'])[idx], dtype=torch.float32, device=device)
        F = np.load(WORK / f'fields_{st}.npz')
        z = F['z'][idx]
        dz = np.abs(z[:, :-1] - z[:, 1:])
        mid = lambda a: 0.5 * (a[:, :-1] + a[:, 1:])
        hyd = np.stack([mid(F['clwc'][idx]) * dz, mid(F['crwc'][idx]) * dz,
                        mid(F['cswc'][idx]) * dz, mid(F['ciwc'][idx]) * dz], axis=-1).astype(np.float32)
        sfc = np.stack([prep['t2m'][idx], prep['q2m'][idx], prep['sp'][idx],
                        prep['tclw'][idx], prep['tcrw'][idx]], axis=-1).astype(np.float32)
        self.s = torch.tensor((sfc - sc['smn'][None]) / sc['sr'][None], dtype=torch.float32, device=device)
        self.h = torch.tensor((hyd - sc['hmn'][None]) / sc['hr'][None], dtype=torch.float32, device=device)
        self.model = cls(**kwargs).to(device).eval()
        ck = torch.load(MODEL_DIR / checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ck['model_state_dict'])
        self.thr_amn = torch.tensor(sc['amn'], dtype=torch.float32, device=device)
        self.thr_ar = torch.tensor(sc['ar'], dtype=torch.float32, device=device)
        self.thr_ymn = torch.tensor(sc['ymn'], dtype=torch.float32, device=device)
        self.thr_yr = torch.tensor(sc['yr'], dtype=torch.float32, device=device)
        self.thr_scT = torch.tensor(sc['scT'], dtype=torch.float32, device=device)
        self.thr_scQ = torch.tensor(sc['scQ'], dtype=torch.float32, device=device)

    @staticmethod
    def _clear(st, idx):
        F = np.load(WORK / f'fields_{st}.npz')
        return np.maximum.reduce([F['clwc'][idx].max(1), F['ciwc'][idx].max(1),
                                  F['cswc'][idx].max(1), F['crwc'][idx].max(1)]) <= 1e-8

    def _atm(self, states, lo):
        T = torch.as_tensor(states[:, :NLEV], dtype=torch.float32, device=self.device)
        q = torch.as_tensor(states[:, NLEV:], dtype=torch.float32, device=self.device)
        atm = torch.stack([q, T, self.p[lo:lo + states.shape[0]]], -1)
        return (atm - self.thr_amn[None]) / self.thr_ar[None]

    def forward(self, states, batch=None):
        batch = batch or self.batch
        M = states.shape[0]
        out = np.empty((M, NCHAN), np.float32)
        ci = np.where(self.clear)[0]
        yi = np.where(~self.clear)[0]
        with torch.inference_mode():
            for lo in range(0, M, batch):
                hi = min(lo + batch, M)
                am = self._atm(states[lo:hi], lo)
                cc = ci[(ci >= lo) & (ci < hi)] - lo
                yy = yi[(yi >= lo) & (yi < hi)] - lo
                if len(cc):
                    o = self.model(am[cc], self.s[lo + cc])
                    out[lo + cc] = (o * self.thr_yr + self.thr_ymn).cpu().numpy()
                if len(yy):
                    o = self.model(am[yy], self.s[lo + yy], self.h[lo + yy])
                    out[lo + yy] = (o * self.thr_yr + self.thr_ymn).cpu().numpy()
        return out

    def jacobian(self, states, batch=None):
        if self.name == 'BiLSTM':
            return self._jacobian_fd(states, batch=batch)
        batch = batch or self.batch
        M = states.shape[0]
        K = np.zeros((M, NCHAN, NS), np.float32)
        ci = np.where(self.clear)[0]
        yi = np.where(~self.clear)[0]
        for lo in range(0, M, batch):
            hi = min(lo + batch, M)
            am = self._atm(states[lo:hi], lo)
            cc = ci[(ci >= lo) & (ci < hi)] - lo
            yy = yi[(yi >= lo) & (yi < hi)] - lo
            if len(cc):
                a = am[cc].detach().clone().requires_grad_(True)
                s = self.s[lo + cc]
                fn = lambda x, sv: self.model(x.unsqueeze(0), sv.unsqueeze(0)).squeeze(0)
                J = torch.func.vmap(torch.func.jacrev(fn, argnums=0), in_dims=(0, 0))(a, s)
                J = J.detach().cpu().numpy()
                K[lo + cc, :, :NLEV] = J[:, :, :, 1] * self.sc['scT'][None]
                K[lo + cc, :, NLEV:] = J[:, :, :, 0] * self.sc['scQ'][None]
            if len(yy):
                a = am[yy].detach().clone().requires_grad_(True)
                s = self.s[lo + yy]
                h = self.h[lo + yy]
                fn = lambda x, sv, hv: self.model(x.unsqueeze(0), sv.unsqueeze(0), hv.unsqueeze(0)).squeeze(0)
                J = torch.func.vmap(torch.func.jacrev(fn, argnums=0), in_dims=(0, 0, 0))(a, s, h)
                J = J.detach().cpu().numpy()
                K[lo + yy, :, :NLEV] = J[:, :, :, 1] * self.sc['scT'][None]
                K[lo + yy, :, NLEV:] = J[:, :, :, 0] * self.sc['scQ'][None]
        return K

    def _jacobian_fd(self, states, eps_T=0.05, eps_q=0.01, batch=None):
        batch = batch or self.batch
        M = states.shape[0]
        y0 = self.forward(states, batch=batch)
        K = np.zeros((M, NCHAN, NS), np.float32)
        for level in range(NLEV):
            Xp = states.copy()
            Xp[:, level] += eps_T
            K[:, :, level] = (self.forward(Xp, batch=batch) - y0) / eps_T
            Xq = states.copy()
            Xq[:, NLEV + level] += eps_q
            K[:, :, NLEV + level] = (self.forward(Xq, batch=batch) - y0) / eps_q
        return K


def lm_run(ev, xb, obs, yb, Binv, RinvP, iters, capT, capQ):
    M = obs.shape[0]
    CAP = np.r_[np.full(NLEV, capT), np.full(NLEV, capQ)]
    X = np.repeat(xb[None], M, 0).astype(np.float64)
    lam = np.full(M, 1e-2)
    def cost(Xv, Yv):
        dx = Xv - xb[None]
        return 0.5 * np.einsum('ni,ij,nj->n', dx, Binv, dx) + 0.5 * np.sum((obs - Yv) ** 2 * RinvP, 1)
    t0 = time.perf_counter()
    for it in range(iters):
        Y = ev.forward(X)
        K = ev.jacobian(X)
        g = np.einsum('ij,nj->ni', Binv, X - xb[None]) - np.einsum('nij,nj->ni', K.transpose(0, 2, 1) * RinvP[:, None, :], obs - Y)
        KtR = K.transpose(0, 2, 1) * RinvP[:, None, :]
        H0 = Binv[None] + np.einsum('nij,njk->nik', KtR, K) + 1e-6 * np.eye(NS)[None]
        Jc = cost(X, Y)
        done = np.zeros(M, bool)
        for _ in range(6):
            if done.all():
                break
            H = H0 + lam[:, None, None] * np.eye(NS)[None]
            dX = np.clip(-np.linalg.solve(H, g[..., None])[..., 0], -CAP, CAP)
            Xn = np.clip(X + dX, CLO, CHI)
            Yn = ev.forward(Xn)
            Jn = cost(Xn, Yn)
            take = (~done) & (Jn < Jc)
            X[take] = Xn[take]; Y[take] = Yn[take]; Jc[take] = Jn[take]
            lam[take] = np.maximum(lam[take] / 3, 1e-6); lam[~take] *= 10
            done |= take
        print(f'    {ev.name} iteration {it + 1}/{iters} done in {time.perf_counter() - t0:.1f}s', flush=True)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations', default='50953,52203,59097,58238,54623')
    ap.add_argument('--models', default='Transformer,MLP,BiLSTM')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--threads', type=int, default=25)
    ap.add_argument('--out', default=str(Path(os.environ.get('QJRMS_OUT_DIR', _QJRMS_ROOT / 'outputs')) / 'cpu_retrieval_timing.csv'))
    args = ap.parse_args()
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    sc = load_scalers()
    stations = [int(x) for x in args.stations.split(',') if x.strip()]
    model_names = [x.strip() for x in args.models.split(',') if x.strip()]
    rows = []
    for st in stations:
        prep = dict(np.load(WORK / f'prep_{st}.npz', allow_pickle=True))
        idx = np.where(prep['keep'])[0]
        xb = np.asarray(prep['xb'], np.float64)
        B = np.asarray(prep['B'], np.float64)
        Binv = np.linalg.inv(B)
        obs = prep['obs'][idx].astype(np.float64)
        yb = prep['Yb'][idx].astype(np.float64)
        Rprof = 0.5 ** 2 + (prep['b_c'][None, :] ** 2 + (prep['a_c'][None, :] * np.minimum(prep['tclw'][idx], prep['clip'])[:, None]) ** 2) + 1.0 ** 2
        RinvP = (1.0 / Rprof).astype(np.float64)
        print(f'STATION {st}: profiles={len(idx)} clear={int(prep["clear"][idx].sum())}', flush=True)
        for name in model_names:
            ev = Evaluator(name, device, sc, st, idx, prep, batch=256 if name != 'BiLSTM' else 2048)
            print(f'  model {name}', flush=True)
            seconds = lm_run(ev, xb, obs, yb, Binv, RinvP, args.iters, 10.0, 2.0)
            rows.append({'station': st, 'model': name, 'profiles': len(idx), 'threads': args.threads,
                         'iters': args.iters, 'seconds': seconds, 'seconds_per_profile': seconds / len(idx)})
            print(f'  {name} total {seconds:.1f}s', flush=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
