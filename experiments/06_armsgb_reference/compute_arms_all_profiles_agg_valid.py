import os
from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

import numpy as np, joblib
from pathlib import Path
from netCDF4 import Dataset
JROOT=Path(os.environ.get('QJRMS_JMATRIX_DIR', _QJRMS_ROOT / 'data' / 'Jmatrix'))
DATA=Path(os.environ.get('QJRMS_DATA_DIR', _QJRMS_ROOT / 'data'))
STATS=joblib.load(os.environ.get('QJRMS_JACOBIAN_STATS', _QJRMS_ROOT / 'model_assets' / 'jacobian_stats.pkl'))
OUT=Path(os.environ.get('QJRMS_OUT_DIR', _QJRMS_ROOT / 'outputs')) / 'arms_all_profiles_aggregate_valid.npz'
STATIONS=[50953,52203,59097,58238,54623]
sums={v:np.zeros((22,37 if v in ('q','t') else 36),np.float64) for v in ['q','t','clwc','crwc']}
sqs={v:np.zeros_like(x) for v,x in sums.items()}
count=0
for st in STATIONS:
    d=np.load(DATA/f'{st}.npz',allow_pickle=True); qp=d['q'].astype(float); n=len(qp)
    print('station',st,n,flush=True)
    with Dataset(JROOT/f'J_{st}.nc') as ds:
        for i in range(0,n,256):
            j=min(i+256,n); b=j-i
            jt=np.transpose(np.asarray(ds.variables['jacobian_temperature'][:,:,i:j],np.float64),(2,0,1))
            jq=np.transpose(np.asarray(ds.variables['jacobian_h2o'][:,:,i:j],np.float64),(2,0,1))
            jc=np.transpose(np.asarray(ds.variables['jacobian_cloud'][:,:,:,i:j],np.float64),(3,1,2,0))
            finite=np.isfinite(jq).all(axis=(1,2)) & np.isfinite(jt).all(axis=(1,2)) & np.isfinite(jc).all(axis=(1,2,3))
            if not finite.all(): print('  skip',int((~finite).sum()),'nonfinite profiles in chunk',i,flush=True)
            jq=jq[finite]; jt=jt[finite]; jc=jc[finite]; qpv=qp[i:j][finite]
            for ch in range(22):
                qc=np.clip(jq[:,ch,:],STATS['atm_min'][ch,:,0],STATS['atm_max'][ch,:,0])*qpv
                tc=np.clip(jt[:,ch,:],STATS['atm_min'][ch,:,1],STATS['atm_max'][ch,:,1])
                cwc=np.clip(jc[:,ch,:,0],STATS['cloud_min'][ch,:,0],STATS['cloud_max'][ch,:,0])
                rwc=np.clip(jc[:,ch,:,1],STATS['cloud_min'][ch,:,1],STATS['cloud_max'][ch,:,1])
                for var,arr in [('q',qc),('t',tc),('clwc',cwc),('crwc',rwc)]:
                    sums[var][ch]+=arr.sum(0); sqs[var][ch]+=(arr*arr).sum(0)
            count+=int(finite.sum())
            if i%1024==0: print(' ',i,'/',n,flush=True)
np.savez_compressed(OUT,count=np.array(count),**{f'{v}_sum':sums[v] for v in sums},**{f'{v}_sq':sqs[v] for v in sums})
print('saved',OUT,count)
