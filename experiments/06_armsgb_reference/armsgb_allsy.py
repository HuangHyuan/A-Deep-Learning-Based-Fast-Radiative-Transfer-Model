from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""All-sky ARMS forward/Jacobian runner (text driver with hydrometeors).
Same interface shape as mwrkit.ARMSgbRunner but writes the extended input file."""
import os, subprocess, tempfile
import numpy as np

class ARMSgbAllSky:
    MAGIC = 'ARMSGB'
    def __init__(self, exe, freq_ghz, zenith_deg=0.0, workdir=None, timeout=300):
        self.exe = exe; self.freq = np.asarray(freq_ghz, float); self.nchan = self.freq.size
        self.zenith = float(zenith_deg); self.workdir = workdir or tempfile.mkdtemp(prefix='allsy_')
        self.timeout = timeout
        os.makedirs(self.workdir, exist_ok=True)

    def _write(self, path, p, t, q, psfc, t2m, q2m, o3, z, clw, ciw, csw, crw, want_jac):
        with open(path, 'w') as f:
            f.write(self.MAGIC + '\n')
            f.write('%d\n' % p.size)
            f.write(' '.join('%.10e' % v for v in p) + '\n')
            f.write(' '.join('%.10e' % v for v in t) + '\n')
            f.write(' '.join('%.10e' % v for v in q) + '\n')
            f.write('%d\n' % self.nchan)
            f.write(' '.join('%.4f' % v for v in self.freq) + '\n')
            f.write('%.4f\n' % self.zenith)
            f.write('%.10e %.10e %.10e\n' % (psfc, t2m, q2m))
            f.write('%d\n' % int(want_jac))
            f.write(' '.join('%.10e' % v for v in o3) + '\n')
            f.write(' '.join('%.10e' % v for v in z) + '\n')
            for a in (clw, crw, csw, ciw):
                f.write(' '.join('%.10e' % v for v in a) + '\n')

    def _call(self, path, out):
        r = subprocess.run([self.exe, path, out], capture_output=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError('driver failed: ' + r.stderr.decode(errors='ignore')[:300])
        tok = open(out).read().split()
        n = int(float(tok[0]))
        tb = np.array(tok[1:1+n], float)
        if tb.size != self.nchan:
            raise RuntimeError('driver returned %d channels, expected %d' % (tb.size, self.nchan))
        i = 1 + n
        K = None
        if i < len(tok):
            nc = int(float(tok[i])); ncol = int(float(tok[i+1]))
            K = np.array(tok[i+2:i+2+nc*ncol], float).reshape(nc, ncol)
        return tb, K

    def forward_jac(self, p, t, q, psfc, t2m, q2m, o3, z, clw, ciw, csw, crw, need_jac=False):
        fin = os.path.join(self.workdir, 'profile_%d.in' % os.getpid())
        fout = os.path.join(self.workdir, 'tb_%d.out' % os.getpid())
        self._write(fin, p, t, q, psfc, t2m, q2m, o3, z, clw, ciw, csw, crw, need_jac)
        return self._call(fin, fout)
