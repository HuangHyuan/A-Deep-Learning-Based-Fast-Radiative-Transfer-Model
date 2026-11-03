import os
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

"""Compare the existing 1D-Var products with matched radiosonde profiles."""
import argparse, csv
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

WORK = Path(os.environ.get('QJRMS_RETRIEVAL_WORK', _QJRMS_ROOT / 'data' / 'retrieval_work'))
RES = Path(os.environ.get('QJRMS_OUT_DIR', _QJRMS_ROOT / 'outputs')) / 'retrieval'
NLEV = 37
MODELS = [
    ('ARMS', '#2E86C1', ':', 'o', 2.3, 4.0),
    ('Transformer', '#C0392B', '--', 's', 1.8, 4.0),
    ('MLP', '#1E8449', '-.', '^', 1.8, 4.0),
    ('BiLSTM', '#7D3C98', (0, (6, 2)), 'D', 1.8, 3.7),
]

def load_case(st, npz):
    with Dataset(WORK/'Dataset'/'REAL'/('%d_obs_era5_202501_202506.nc' % st)) as ds:
        dates = np.array([str(x) for x in ds.variables['Date_Time'][:]])
        era5raw = np.concatenate([np.asarray(ds.variables['Level_Temperature'][:], float).T,
                                  np.asarray(ds.variables['Level_H2O'][:], float).T], 1)
        obs = np.asarray(ds.variables['Obs_BT'][:], float).T
    ok = np.isfinite(era5raw).all(1) & np.isfinite(obs).all(1) & (obs > 0).all(1) & (obs <= 330).all(1)
    prep = dict(np.load(WORK/('prep_%d.npz' % st), allow_pickle=True))
    F = np.load(WORK/('fields_%d.npz' % st))
    files = {name: np.load(RES/('result_%d_%s.npz' % (st, name))) for name, *_ in MODELS}
    idx = files['ARMS']['idx']
    for name in files: assert np.array_equal(idx, files[name]['idx'])
    S = np.load(npz)
    times = np.array(dates)[ok][idx]
    def todt(x):
        x = str(x); return np.datetime64(x[:4]+'-'+x[4:6]+'-'+x[6:8]+'T'+x[8:10]+':'+x[10:12]+':'+x[12:14])
    obs_t = np.array([todt(x) for x in times])
    snd_t = np.array([todt(x) for x in S['time_actual']])
    pairs = []
    for j, t in enumerate(snd_t):
        dm = np.abs((obs_t-t).astype('timedelta64[m]').astype(int))
        i = int(np.argmin(dm))
        if dm[i] <= 30: pairs.append((i, j))
    return dict(prep=prep, F=F, files=files, S=S, S0=S, idx=idx, times=times,
                p=F['p'][idx], z=F['z'][idx]/1000.0, era5=prep['era5'][idx],
                tr=prep['tr'][idx], te=prep['te'][idx], pairs=pairs)

def interp_profile(ps, vs, pt):
    ps = np.asarray(ps, float); vs = np.asarray(vs, float)
    good = np.isfinite(ps) & np.isfinite(vs) & (ps > 0)
    ps = ps[good]; vs = vs[good]
    order = np.argsort(ps); ps = ps[order]; vs = vs[order]
    out = np.full(len(pt), np.nan)
    valid = (pt >= ps[0]) & (pt <= ps[-1])
    out[valid] = np.interp(np.log(pt[valid]), np.log(ps), vs)
    return out

def prepare_truth(c, pairs):
    S = c['S']; p = c['p']; out = {}
    for i, j in pairs:
        s0 = int(S['start'][j]); n = int(S['count'][j])
        ps = S['p'][s0:s0+n]
        out[i] = (interp_profile(ps, S['T'][s0:s0+n], p[i]),
                  interp_profile(ps, S['q'][s0:s0+n], p[i]))
    return out

def metrics(x, c, pairs, truth, mask):
    rows = {'T': [], 'q': []}
    for i, j in pairs:
        Tt, qt = truth[i]; good = mask[i] & np.isfinite(Tt) & np.isfinite(qt)
        if good.sum() >= 3:
            rows['T'].append(np.sqrt(np.mean((x[i, :NLEV][good]-Tt[good])**2)))
            rows['q'].append(np.sqrt(np.mean((x[i, NLEV:][good]-qt[good])**2)))
    return float(np.median(rows['T'])) if rows['T'] else np.nan, float(np.median(rows['q'])) if rows['q'] else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--station', type=int, default=int(os.environ.get('QJRMS_SOUNDING_STATION', '58238')))
    ap.add_argument('--period', choices=['all', 'test', 'train'], default='all')
    ap.add_argument('--npz', default=str(WORK / 'sounding_58238_profiles.npz'))
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    c = load_case(a.station, a.npz)
    if a.period == 'test': pairs = [(i, j) for i, j in c['pairs'] if c['te'][i]]
    elif a.period == 'train': pairs = [(i, j) for i, j in c['pairs'] if c['tr'][i]]
    else: pairs = c['pairs']
    truth = prepare_truth(c, pairs)
    z = c['z']; zm = z.mean(0); xb = c['prep']['xb']
    refs = {'Background': np.repeat(xb[None], len(c['idx']), 0), 'ERA5': c['era5']}
    refs.update({name: f['xa'] for name, f in c['files'].items()})
    regions = {'all': (z >= 0) & (z <= 20), '0-2km': (z >= 0) & (z < 2),
               '2-5km': (z >= 2) & (z < 5), '5-10km': (z >= 5) & (z < 10),
               '10-20km': (z >= 10) & (z < 20)}
    tag = a.tag or ('%d_%s' % (a.station, a.period)); outdir = RES/'sounding'; outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/('sounding_metrics_%s.csv' % tag), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'region', 'n', 'T_RMSE_K', 'q_RMSE_gkg'])
        for name, x in refs.items():
            for reg, mask in regions.items():
                T, q = metrics(x, c, pairs, truth, mask)
                n = sum(1 for i, j in pairs if (mask[i] & np.isfinite(truth[i][0]) & np.isfinite(truth[i][1])).sum() >= 3)
                w.writerow([name, reg, n, '%.6f' % T, '%.6f' % q])
                print('%-12s %-7s n=%3d T=%.3f q=%.3f' % (name, reg, n, T, q), flush=True)

if __name__ == '__main__': main()
