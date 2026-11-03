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

"""Prepare Chinese radiosonde CSV files into a compact QC'ed sounding NPZ."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--station', default=os.environ.get('QJRMS_SOUNDING_STATION', '58238'))
    ap.add_argument('--tk-root', default=os.environ.get('QJRMS_TK_ROOT', str(_QJRMS_ROOT / 'data' / 'TK')))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    fs = sorted(Path(a.tk_root, a.station).glob('*.csv'))
    if not fs: raise SystemExit('no CSV files under %s' % Path(a.tk_root, a.station))
    parts = []
    cols = ['Datetime','Year_Data','Mon_Data','Day_Data','Hour_Data','Min','Second',
            'PRS_HWC','GPH','TEM','RHU','Q_PRS_HWC','Q_GPH','Q_TEM','Q_RHU']
    for f in fs:
        d = pd.read_csv(f, encoding='gb18030', usecols=cols).mask(lambda x: x == 999999)
        d['Datetime'] = pd.to_datetime(d.Datetime)
        good = (d.PRS_HWC.notna() & d.TEM.notna() & d.RHU.notna() &
                (d.Q_PRS_HWC == 0) & (d.Q_TEM == 0) & (d.Q_RHU == 0) &
                (d.PRS_HWC >= 1) & (d.PRS_HWC <= 1100) &
                (d.TEM > -120) & (d.TEM < 60) & (d.RHU >= 0) & (d.RHU <= 110))
        parts.append(d.loc[good])
    d = pd.concat(parts, ignore_index=True)
    times0, timesa, P, T, Q, Z, starts, counts = [], [], [], [], [], [], [], []
    for t, g0 in d.groupby('Datetime'):
        r0 = g0.iloc[0]
        actual = pd.Timestamp(year=int(r0.Year_Data), month=int(r0.Mon_Data), day=int(r0.Day_Data),
                              hour=int(r0.Hour_Data), minute=int(r0.Min), second=int(r0.Second))
        g = g0.sort_values('PRS_HWC', ascending=False)
        g = g.groupby('PRS_HWC', as_index=False).agg({'TEM':'mean','RHU':'mean','GPH':'mean'})
        g = g.sort_values('PRS_HWC', ascending=False)
        p = g.PRS_HWC.to_numpy(float); tc = g.TEM.to_numpy(float); rh = g.RHU.to_numpy(float); z = g.GPH.to_numpy(float)
        ew = 6.112*np.exp(17.67*tc/(tc+243.5)); ei = 6.112*np.exp(22.46*tc/(tc+272.62))
        es = np.where(tc >= 0, ew, ei); e = np.clip(rh/100.0, 0, 1.1)*es
        q = 622.0*e/(p-0.378*e)
        starts.append(len(P)); counts.append(len(p))
        times0.append(pd.Timestamp(t).strftime('%Y%m%d%H%M%S'))
        timesa.append(actual.strftime('%Y%m%d%H%M%S'))
        P.extend(p); T.extend(tc+273.15); Q.extend(q); Z.extend(z)
    out = Path(a.out) if a.out else Path('sounding_%s_profiles.npz' % a.station)
    np.savez_compressed(out, time_nominal=np.array(times0), time_actual=np.array(timesa),
                        start=np.array(starts), count=np.array(counts), p=np.array(P, np.float32),
                        T=np.array(T, np.float32), q=np.array(Q, np.float32), z=np.array(Z, np.float32))
    print('saved', out, 'profiles', len(times0), 'levels', len(P))

if __name__ == '__main__': main()
