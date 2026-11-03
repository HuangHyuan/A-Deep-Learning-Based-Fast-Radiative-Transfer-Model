from pathlib import Path
import sys
import os
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

import sys, time
from pathlib import Path
import numpy as np
import torch
import joblib
from netCDF4 import Dataset

PROJECT = _QJRMS_ROOT
DATA = Path(os.environ.get('QJRMS_DATA_DIR', _QJRMS_ROOT / 'data'))
JROOT = Path(os.environ.get('QJRMS_JMATRIX_DIR', _QJRMS_ROOT / 'data' / 'Jmatrix'))
MODEL_DIR = Path(os.environ.get('QJRMS_MODEL_DIR', _QJRMS_ROOT / 'model_assets'))
sys.path.insert(0, str(_QJRMS_CORE))
from GB_model import UnifiedAllSkyRTM_Transformer, UnifiedAllSkyRTM_MLP, UnifiedAllSkyRTM_BiLSTM
STATIONS = [50953, 52203, 59097, 58238, 54623]
N = 100
SEED = 2026
OUT = Path(os.environ.get('QJRMS_OUT_DIR', _QJRMS_ROOT / 'outputs')) / 'five_station_jacobian_full_100.npz'
torch.set_num_threads(8)


def load_scalers():
    a = joblib.load(MODEL_DIR / 'scaler_atm_global.pkl')
    s = joblib.load(MODEL_DIR / 'scaler_sfc_global.pkl')
    h = joblib.load(MODEL_DIR / 'scaler_hydro_cloudy.pkl')
    y = joblib.load(MODEL_DIR / 'scaler_y.pkl')
    return dict(
        amin=np.array([z.data_min_ for z in a], np.float32), amax=np.array([z.data_max_ for z in a], np.float32),
        hmin=np.array([z.data_min_ for z in h], np.float32), hmax=np.array([z.data_max_ for z in h], np.float32),
        smin=np.asarray(s.data_min_, np.float32), smax=np.asarray(s.data_max_, np.float32),
        ymin=np.asarray(y.data_min_, np.float32), ymax=np.asarray(y.data_max_, np.float32),
    )


def load_model(name):
    if name == 'transformer':
        cls, kw, fn = UnifiedAllSkyRTM_Transformer, dict(hidden_dim=160, nhead=4, num_transformer_layers=3), 'unified_model_jacobian_finetuned.pth'
    elif name == 'mlp':
        cls, kw, fn = UnifiedAllSkyRTM_MLP, dict(hidden_dim=12, num_mlp_blocks=3), 'mlp.pth'
    else:
        cls, kw, fn = UnifiedAllSkyRTM_BiLSTM, dict(hidden_dim=180, num_bi_lstm_blocks=2), 'bilstm.pth'
    ck = torch.load(MODEL_DIR / fn, map_location='cpu', weights_only=False)
    m = cls(**kw)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m


def build(d):
    atm = np.stack([d['q'], d['T'], d['p']], -1).astype(np.float32)
    sfc = np.stack([d['t2m'], d['q2m'], d['psfc'], d['clwc'].sum(1), d['crwc'].sum(1)], -1).astype(np.float32)
    z = d['z'].astype(np.float32)
    dz = np.abs(z[:, :-1] - z[:, 1:])
    lc = lambda a: 0.5 * (a[:, :-1] + a[:, 1:])
    hyd = np.stack([lc(d['clwc']) * dz, lc(d['crwc']) * dz, lc(d['cswc']) * dz, lc(d['ciwc']) * dz], -1).astype(np.float32)
    return atm, sfc, hyd


def compute_jac(model, name, am, sm, hm):
    a = torch.tensor(am)
    s = torch.tensor(sm)
    h = torch.tensor(hm)
    def f(aa, ss, hh):
        return model(aa.unsqueeze(0), ss.unsqueeze(0), hh.unsqueeze(0)).squeeze(0)
    ja_out, jh_out = [], []
    if name == 'bilstm':
        for i in range(len(a)):
            ja = torch.autograd.functional.jacobian(lambda aa: f(aa, s[i], h[i]), a[i], vectorize=False)
            jh = torch.autograd.functional.jacobian(lambda hh: f(a[i], s[i], hh), h[i], vectorize=False)
            ja_out.append(ja.detach().numpy())
            jh_out.append(jh.detach().numpy())
            if (i + 1) % 20 == 0:
                print('  bilstm', i + 1, '/', len(a), flush=True)
        return np.stack(ja_out), np.stack(jh_out)
    for i in range(0, len(a), 10):
        j = min(i + 10, len(a))
        aa, ss, hh = a[i:j], s[i:j], h[i:j]
        ja_out.append(torch.func.vmap(torch.func.jacrev(f, argnums=0), in_dims=(0, 0, 0))(aa, ss, hh).detach().numpy())
        jh_out.append(torch.func.vmap(torch.func.jacrev(f, argnums=2), in_dims=(0, 0, 0))(aa, ss, hh).detach().numpy())
    return np.concatenate(ja_out), np.concatenate(jh_out)


def main():
    sc = load_scalers()
    ar = np.where(sc['amax'] - sc['amin'] > 0, sc['amax'] - sc['amin'], 1).astype(np.float32)
    sr = np.where(sc['smax'] - sc['smin'] > 0, sc['smax'] - sc['smin'], 1).astype(np.float32)
    hr = np.where(sc['hmax'] - sc['hmin'] > 0, sc['hmax'] - sc['hmin'], 1e-30).astype(np.float32)
    yr = (sc['ymax'] - sc['ymin']).astype(np.float32)
    rng = np.random.default_rng(SEED)
    result = {}
    models = {name: load_model(name) for name in ['transformer', 'bilstm', 'mlp']}
    for st in STATIONS:
        print('=== station', st, flush=True)
        d0 = dict(np.load(DATA / f'{st}.npz'))
        n_all = len(d0['q'])
        valid = np.asarray(d0.get('valid', np.ones(n_all))).reshape(-1) > 0
        idx = np.sort(rng.choice(np.where(valid)[0], size=min(N, int(valid.sum())), replace=False))
        d = {k: (np.asarray(v)[idx] if np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == n_all else np.asarray(v)) for k, v in d0.items()}
        atm, sfc, hyd = build(d)
        am = ((atm - sc['amin'][None]) / ar[None]).astype(np.float32)
        sm = ((sfc - sc['smin'][None]) / sr[None]).astype(np.float32)
        hm = ((hyd - sc['hmin'][None]) / hr[None]).astype(np.float32)
        qp = atm[:, :, 0]
        result[f'{st}_pressure'] = atm[:, :, 2].mean(0)
        result[f'{st}_q_profile'] = qp
        with Dataset(JROOT / f'J_{st}.nc') as ds:
            jt = np.transpose(np.asarray(ds.variables['jacobian_temperature'][:,:,idx], dtype=np.float32), (2,0,1))
            jq = np.transpose(np.asarray(ds.variables['jacobian_h2o'][:,:,idx], dtype=np.float32), (2,0,1))
            jc = np.transpose(np.asarray(ds.variables['jacobian_cloud'][:,:,:,idx], dtype=np.float32), (3,1,2,0))
        result[f'{st}_qraw_truth'] = jq
        result[f'{st}_q_truth'] = jq * qp[:, None, :]
        result[f'{st}_t_truth'] = jt
        result[f'{st}_clwc_truth'] = jc[:,:,:,0]
        result[f'{st}_crwc_truth'] = jc[:,:,:,1]
        for name, m in models.items():
            print(' ', name, flush=True)
            t0 = time.time()
            ja, jh = compute_jac(m, name, am, sm, hm)
            qraw = ja[:, :, :, 0] * (yr[None, :, None] / ar[None, None, :, 0])
            t = ja[:, :, :, 1] * (yr[None, :, None] / ar[None, None, :, 1])
            hydro = jh * (yr[None, :, None, None] / hr[None, None, :, :])
            result[f'{st}_qraw_{name}'] = qraw.astype(np.float32)
            result[f'{st}_q_{name}'] = (qraw * qp[:, None, :]).astype(np.float32)
            result[f'{st}_t_{name}'] = t.astype(np.float32)
            result[f'{st}_clwc_{name}'] = hydro[:,:,:,0].astype(np.float32)
            result[f'{st}_crwc_{name}'] = hydro[:,:,:,1].astype(np.float32)
            print('  done', round(time.time()-t0,1), 's', flush=True)
    np.savez_compressed(OUT, **result)
    print('saved', OUT)


if __name__ == '__main__':
    main()
