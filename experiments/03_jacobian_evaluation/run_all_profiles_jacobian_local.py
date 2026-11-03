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

import os, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import torch
import joblib

DATA=Path(os.environ.get('QJRMS_DATA_DIR', _QJRMS_ROOT / 'data'))
MODEL_DIR=Path(os.environ.get('QJRMS_MODEL_DIR', _QJRMS_ROOT / 'model_assets'))
STATS_PATH=Path(os.environ.get('QJRMS_JACOBIAN_STATS', _QJRMS_ROOT / 'model_assets' / 'jacobian_stats.pkl'))
OUT=Path(os.environ.get('QJRMS_OUT_DIR', _QJRMS_ROOT / 'outputs')) / 'all_profiles_jacobian_final_correct'
OUT.mkdir(parents=True,exist_ok=True)
STATIONS=[50953,52203,59097,58238,54623]
if len(sys.argv)>1: STATIONS=[int(x) for x in sys.argv[1:]]
MODELS=['transformer','bilstm','mlp']
SEL=[0,6,10,16]


def worker(st):
    os.environ['OMP_NUM_THREADS']='4'; torch.set_num_threads(4)
    sys.path.insert(0,str(MODEL_DIR))
    from GB_model import UnifiedAllSkyRTM_Transformer,UnifiedAllSkyRTM_MLP,UnifiedAllSkyRTM_BiLSTM
    a=joblib.load(MODEL_DIR/'scaler_atm_global.pkl'); s=joblib.load(MODEL_DIR/'scaler_sfc_global.pkl')
    h=joblib.load(MODEL_DIR/'scaler_hydro_cloudy.pkl'); y=joblib.load(MODEL_DIR/'scaler_y.pkl'); sts=joblib.load(STATS_PATH)
    amin=np.array([z.data_min_ for z in a],np.float32); amax=np.array([z.data_max_ for z in a],np.float32)
    hmin=np.array([z.data_min_ for z in h],np.float32); hmax=np.array([z.data_max_ for z in h],np.float32)
    smin=np.asarray(s.data_min_,np.float32); smax=np.asarray(s.data_max_,np.float32)
    ymin=np.asarray(y.data_min_,np.float32); ymax=np.asarray(y.data_max_,np.float32)
    ar=np.where(amax-amin>0,amax-amin,1).astype(np.float32); sr=np.where(smax-smin>0,smax-smin,1).astype(np.float32)
    hr=np.where(hmax-hmin>0,hmax-hmin,1e-30).astype(np.float32); yr=(ymax-ymin).astype(np.float32)
    d=dict(np.load(DATA/f'{st}.npz',allow_pickle=True)); n=len(d['q'])
    bad_q2m=~np.isfinite(np.asarray(d['q2m'],dtype=np.float64))
    if bad_q2m.any():
        d['q2m']=np.asarray(d['q2m'],dtype=np.float32).copy(); d['q2m'][bad_q2m]=np.asarray(d['q'],dtype=np.float32)[bad_q2m,0]
    atm=np.stack([d['q'],d['T'],d['p']],-1).astype(np.float32)
    sfc=np.stack([d['t2m'],d['q2m'],d['psfc'],d['clwc'].sum(1),d['crwc'].sum(1)],-1).astype(np.float32)
    z=d['z'].astype(np.float32); dz=np.abs(z[:,:-1]-z[:,1:]); lc=lambda q:0.5*(q[:,:-1]+q[:,1:])
    hyd=np.stack([lc(d['clwc'])*dz,lc(d['crwc'])*dz,lc(d['cswc'])*dz,lc(d['ciwc'])*dz],-1).astype(np.float32)
    am=((atm-amin[None])/ar[None]).astype(np.float32); sm=((sfc-smin[None])/sr[None]).astype(np.float32); hm=((hyd-hmin[None])/hr[None]).astype(np.float32)
    qp=atm[:,:,0]
    specs={
      'transformer':(UnifiedAllSkyRTM_Transformer,dict(hidden_dim=160,nhead=4,num_transformer_layers=3),'unified_model_jacobian_finetuned.pth'),
      'mlp':(UnifiedAllSkyRTM_MLP,dict(hidden_dim=12,num_mlp_blocks=3),'mlp.pth'),
      'bilstm':(UnifiedAllSkyRTM_BiLSTM,dict(hidden_dim=180,num_bi_lstm_blocks=2),'bilstm.pth')}
    models={}
    for name,(cls,kw,fn) in specs.items():
        ck=torch.load(MODEL_DIR/fn,map_location='cpu',weights_only=False); m=cls(**kw); m.load_state_dict(ck['model_state_dict']); m.eval(); models[name]=m
    acc={}
    for name in MODELS:
        for var in ['q','t','clwc','crwc']:
            acc[f'{var}_sum_{name}']=np.zeros((4,37 if var in ('q','t') else 36),np.float64)
            acc[f'{var}_sq_{name}']=np.zeros_like(acc[f'{var}_sum_{name}'])
    def add(name,ja,jh,idx):
        qraw=ja[:,:,:,0]*(yr[SEL][None,:,None]/ar[None,None,:,0])
        t=ja[:,:,:,1]*(yr[SEL][None,:,None]/ar[None,None,:,1])
        hydro=jh*(yr[SEL][None,:,None,None]/hr[None,None,:,:])
        for oi,ch in enumerate(SEL):
            qc=np.clip(qraw[:,oi,:],sts['atm_min'][ch,:,0],sts['atm_max'][ch,:,0])*qp[idx]
            tc=np.clip(t[:,oi,:],sts['atm_min'][ch,:,1],sts['atm_max'][ch,:,1])
            cwc=np.clip(hydro[:,oi,:,0],sts['cloud_min'][ch,:,0],sts['cloud_max'][ch,:,0])
            rwc=np.clip(hydro[:,oi,:,1],sts['cloud_min'][ch,:,1],sts['cloud_max'][ch,:,1])
            for var,arr in [('q',qc),('t',tc),('clwc',cwc),('crwc',rwc)]:
                acc[f'{var}_sum_{name}'][oi]+=arr.sum(0); acc[f'{var}_sq_{name}'][oi]+=(arr*arr).sum(0)
    bs=8
    for nm,m in models.items():
        t0=time.time(); print(st,nm,'START',flush=True)
        for i in range(0,n,bs):
            j=min(i+bs,n); ai=torch.tensor(am[i:j]); si=torch.tensor(sm[i:j]); hi=torch.tensor(hm[i:j])
            def f(aa,ss,hh): return m(aa.unsqueeze(0),ss.unsqueeze(0),hh.unsqueeze(0)).squeeze(0)[SEL]
            ja=torch.func.vmap(torch.func.jacrev(f,argnums=0),in_dims=(0,0,0))(ai,si,hi).detach().numpy()
            jh=torch.func.vmap(torch.func.jacrev(f,argnums=2),in_dims=(0,0,0))(ai,si,hi).detach().numpy()
            add(nm,ja,jh,slice(i,j))
            if i%400==0: print(st,nm,i,'/',n,flush=True)
        print(st,nm,'DONE',round(time.time()-t0,1),'s',flush=True)
    acc['count']=np.array(n); acc['channels']=np.array([1,7,11,17])
    np.savez_compressed(OUT/f'{st}_agg.npz',**acc)
    print(st,'SAVED',n,flush=True); return st

if __name__=='__main__':
    print('start',flush=True)
    with Pool(len(STATIONS)) as pool:
        for st in pool.imap_unordered(worker,STATIONS): print('finished',st,flush=True)
    print('ALL DONE',flush=True)
