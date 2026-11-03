"""Parallel ARMS evaluation on AutoDL (static ifort driver + multiprocessing pool)."""
import os, subprocess, tempfile
import numpy as np
from multiprocessing import Pool

DRV = os.environ.get('ARMS_DRV', '')
NPZ = os.environ.get('ARMS_FIELDS', '')
_G = {}

def _init():
    z = np.load(NPZ, allow_pickle=True)
    for k in ('p', 'o3', 'z', 'clwc', 'ciwc', 'cswc', 'crwc', 'sp', 't2m', 'q2m', 'freq'):
        _G[k] = z[k]
    _G['tmp'] = tempfile.mkdtemp(prefix='armsw_')

def _write(path, pin, tin, qin, psfc, t2m, q2m, o3, z, clw, ciw, csw, crw, jac):
    with open(path, 'w') as f:
        f.write('ARMSGB\n%d\n' % pin.size)
        f.write(' '.join('%.10e' % v for v in pin) + '\n')
        f.write(' '.join('%.10e' % v for v in tin) + '\n')
        f.write(' '.join('%.10e' % v for v in qin) + '\n')
        f.write('%d\n' % _G['freq'].size)
        f.write(' '.join('%.4f' % v for v in _G['freq']) + '\n')
        f.write('0.0000\n%.10e %.10e %.10e\n%d\n' % (psfc, t2m, q2m, int(jac)))
        f.write(' '.join('%.10e' % v for v in o3) + '\n')
        f.write(' '.join('%.10e' % v for v in z) + '\n')
        for a in (clw, crw, csw, ciw):
            f.write(' '.join('%.10e' % v for v in a) + '\n')

def _one(a):
    i, T, q, jac = a
    p = _G['p'][i]; o3 = _G['o3'][i]; z = _G['z'][i]
    fin = os.path.join(_G['tmp'], 'i.in'); fout = os.path.join(_G['tmp'], 'o.out')
    _write(fin, p, T, q, float(_G['sp'][i]), float(_G['t2m'][i]), float(_G['q2m'][i]),
           o3, z, _G['clwc'][i], _G['ciwc'][i], _G['cswc'][i], _G['crwc'][i], jac)
    r = subprocess.run([DRV, fin, fout], capture_output=True)
    if r.returncode != 0:
        return i, None, None
    tok = open(fout).read().split()
    nc = int(float(tok[0])); tb = np.array(tok[1:1+nc], float)
    K = None
    if jac:
        p0 = nc + 1; ncol = int(float(tok[p0+1]))
        K = np.array(tok[p0+2:p0+2+nc*ncol], float).reshape(nc, ncol)
    return i, tb, K

class ArmsEvaluator:
    def __init__(self, nproc=32):
        self.pool = Pool(nproc, initializer=_init)
        self.NPZ = NPZ
    def __call__(self, states, idx, need_jac):
        tasks = [(int(i), states[k, :37], states[k, 37:], need_jac) for k, i in enumerate(idx)]
        out = self.pool.map(_one, tasks, chunksize=8)
        Y = np.empty((len(idx), 22)); K = np.empty((len(idx), 22, 74)) if need_jac else None
        bad = 0
        for k, (i, tb, kk) in enumerate(out):
            if tb is None or not np.isfinite(tb).all():
                bad += 1; Y[k] = np.nan
                if need_jac: K[k] = np.nan
            else:
                Y[k] = tb
                if need_jac: K[k] = kk
        return Y, K, bad
    def close(self):
        self.pool.close(); self.pool.join()
