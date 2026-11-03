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

"""Legacy one-step retrieval example for the five-station prototype.
data/<st>.npz arrays are (N,37); jac_T/jac_Q (N,22,37); obs/fbg (N,22).
Scalers: atm = list of 37 MinMaxScaler(3); hydro = list of 36 MinMaxScaler(4);
         sfc = MinMaxScaler(5); y = MinMaxScaler(22).
Jacobians: Transformer/MLP -> exact autodiff (vmap+jacrev); BiLSTM -> batched finite difference.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch, joblib

BASE=_QJRMS_ROOT
sys.path.insert(0,str(_QJRMS_CORE))
from GB_model import UnifiedAllSkyRTM_Transformer, UnifiedAllSkyRTM_MLP, UnifiedAllSkyRTM_BiLSTM

DATA=Path(os.environ.get('QJRMS_RETRIEVAL_DATA', BASE / 'data' / 'retrieval')); OUT=Path(os.environ.get('QJRMS_OUT_DIR', BASE / 'outputs')) / 'retrieval'; OUT.mkdir(parents=True,exist_ok=True); MODELS=Path(os.environ.get('QJRMS_MODEL_DIR', BASE / 'model_assets'))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATIONS=[50953,52203,59097,58238,54623]
NAMES=['ARMS','transformer','mlp','bilstm']
LABEL={'ARMS':'ARMS-gb','transformer':'Transformer','mlp':'MLP','bilstm':'BiLSTM'}
print('device',DEVICE,flush=True)

def load_scalers():
    a=joblib.load(MODELS/'scaler_atm_global.pkl'); s=joblib.load(MODELS/'scaler_sfc_global.pkl')
    h=joblib.load(MODELS/'scaler_hydro_cloudy.pkl'); y=joblib.load(MODELS/'scaler_y.pkl')
    amn=np.array([x.data_min_ for x in a],np.float32) if isinstance(a,(list,tuple)) else np.asarray(a.data_min_,np.float32)
    amx=np.array([x.data_max_ for x in a],np.float32) if isinstance(a,(list,tuple)) else np.asarray(a.data_max_,np.float32)
    hmn=np.array([x.data_min_ for x in h],np.float32) if isinstance(h,(list,tuple)) else np.asarray(h.data_min_,np.float32)
    hmx=np.array([x.data_max_ for x in h],np.float32) if isinstance(h,(list,tuple)) else np.asarray(h.data_max_,np.float32)
    return dict(amin=amn,amax=amx,smin=np.asarray(s.data_min_,np.float32),smax=np.asarray(s.data_max_,np.float32),
                hmin=hmn,hmax=hmx,ymin=np.asarray(y.data_min_,np.float32),ymax=np.asarray(y.data_max_,np.float32))
SC=load_scalers()
AR=np.where(SC['amax']-SC['amin']>0,SC['amax']-SC['amin'],1.0).astype(np.float32)
SR=np.where(SC['smax']-SC['smin']>0,SC['smax']-SC['smin'],1.0).astype(np.float32)
HR=np.where(SC['hmax']-SC['hmin']>0,SC['hmax']-SC['hmin'],1e-30).astype(np.float32)
YR=(SC['ymax']-SC['ymin']).astype(np.float32)
SC_T=(YR[:,None]/AR[None,:,1]).astype(np.float32); SC_Q=(YR[:,None]/AR[None,:,0]).astype(np.float32)
print('scaler shapes atm',AR.shape,'sfc',SR.shape,'hyd',HR.shape,'y',YR.shape,flush=True)

def load_model(name):
    if name=='transformer': cls,kw,f=UnifiedAllSkyRTM_Transformer,dict(hidden_dim=160,nhead=4,num_transformer_layers=3),'unified_model_jacobian_finetuned.pth'
    elif name=='mlp': cls,kw,f=UnifiedAllSkyRTM_MLP,dict(hidden_dim=12,num_mlp_blocks=3),'mlp.pth'
    else: cls,kw,f=UnifiedAllSkyRTM_BiLSTM,dict(hidden_dim=180,num_bi_lstm_blocks=2),'bilstm.pth'
    ck=torch.load(MODELS/f,map_location='cpu',weights_only=False)
    m=cls(**kw); m.load_state_dict(ck['model_state_dict']); m.eval().to(DEVICE); return m

def build(d):
    atm=np.stack([d['q'],d['T'],d['p']],-1).astype(np.float32)
    sfc=np.stack([d['t2m'],d['q2m'],d['psfc'],d['clwc'].sum(1),d['crwc'].sum(1)],-1).astype(np.float32)
    z=d['z'].astype(np.float32); dz=np.abs(z[:,:-1]-z[:,1:])
    lc=lambda a:0.5*(a[:,:-1]+a[:,1:])
    hyd=np.stack([lc(d['clwc'])*dz,lc(d['crwc'])*dz,lc(d['cswc'])*dz,lc(d['ciwc'])*dz],-1).astype(np.float32)
    clear=((d['clwc'].sum(1)==0)&(d['ciwc'].sum(1)==0)&(d['cswc'].sum(1)==0)&(d['crwc'].sum(1)==0))
    return atm,sfc,hyd,clear

def scale(atm,sfc,hyd):
    am=((atm-SC['amin'][None])/AR[None]).astype(np.float32)
    sm=((sfc-SC['smin'][None])/SR[None]).astype(np.float32)
    hm=((hyd-SC['hmin'][None])/HR[None]).astype(np.float32)
    return am,sm,hm

def forward(model,am,sm,hm,clear,batch=2048):
    N=am.shape[0]; out=np.empty((N,22),np.float32)
    with torch.no_grad():
        for i in range(0,N,batch):
            j=min(i+batch,N); idx=np.arange(i,j); ci=idx[clear[idx]]; yi=idx[~clear[idx]]
            if ci.size:
                y=model(torch.tensor(am[ci],device=DEVICE),torch.tensor(sm[ci],device=DEVICE)).cpu().numpy()
                out[ci]=y*YR[None]+SC['ymin'][None]
            if yi.size:
                y=model(torch.tensor(am[yi],device=DEVICE),torch.tensor(sm[yi],device=DEVICE),torch.tensor(hm[yi],device=DEVICE)).cpu().numpy()
                out[yi]=y*YR[None]+SC['ymin'][None]
    return out

def unwrap(J):
    Jn=J.detach().float().cpu().numpy()
    return Jn[:,:,:,1]*SC_T, Jn[:,:,:,0]*SC_Q

def jacobian_ad(model,am,sm,hm,clear,batch=64):
    N=am.shape[0]; K=np.zeros((N,22,74),np.float32)
    def fc(a,s): return model(a.unsqueeze(0),s.unsqueeze(0)).squeeze(0)
    def fk(a,s,h): return model(a.unsqueeze(0),s.unsqueeze(0),h.unsqueeze(0)).squeeze(0)
    for i in range(0,N,batch):
        j=min(i+batch,N); idx=np.arange(i,j); ci=idx[clear[idx]]; yi=idx[~clear[idx]]
        if ci.size:
            a=torch.tensor(am[ci],device=DEVICE); s=torch.tensor(sm[ci],device=DEVICE)
            J=torch.func.vmap(torch.func.jacrev(fc,argnums=0),in_dims=(0,0))(a,s)
            Jt,Jq=unwrap(J); K[ci,:,:37]=Jt; K[ci,:,37:]=Jq
        if yi.size:
            a=torch.tensor(am[yi],device=DEVICE); s=torch.tensor(sm[yi],device=DEVICE); h=torch.tensor(hm[yi],device=DEVICE)
            J=torch.func.vmap(torch.func.jacrev(fk,argnums=0),in_dims=(0,0,0))(a,s,h)
            Jt,Jq=unwrap(J); K[yi,:,:37]=Jt; K[yi,:,37:]=Jq
        if (i//batch+1)%25==0: print('   vmap',i,flush=True)
    return K

def jacobian_fd(model,am,sm,hm,clear,eps_T=0.05,eps_q=0.01,batch=2048):
    N=am.shape[0]; y0=forward(model,am,sm,hm,clear,batch); K=np.zeros((N,22,74),np.float32)
    for l in range(37):
        ap=am.copy(); ap[:,l,1]+=eps_T/AR[l,1]
        K[:,:,l]=(forward(model,ap,sm,hm,clear,batch)-y0)/eps_T
        aq=am.copy(); aq[:,l,0]+=eps_q/AR[l,0]
        K[:,:,37+l]=(forward(model,aq,sm,hm,clear,batch)-y0)/eps_q
        if (l+1)%9==0: print('   fd',l+1,flush=True)
    return K

def corr_matrix():
    C=np.eye(74)
    for a,b,L in [(0,37,0.5),(37,74,0.6)]:
        cc=np.arange(b-a); C[a:b,a:b]=np.exp(-0.5*((cc[:,None]-cc[None,:])/L)**2)
    return C

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

C=corr_matrix(); rows=[]; summary={}
for st in STATIONS:
    t0=time.time(); print('==== STATION',st,flush=True)
    d=dict(np.load(DATA/f'{st}.npz'))
    obs=d['obs'].astype(np.float64); fbg=d['fbg'].astype(np.float64)
    N0=obs.shape[0]
    good=(np.isfinite(obs).all(1)&np.isfinite(fbg).all(1))
    if 'valid' in d: good=good&(np.asarray(d['valid']).reshape(-1)>0)
    d={k:(np.asarray(v)[good] if (np.asarray(v).ndim>=1 and np.asarray(v).shape[0]==N0) else np.asarray(v)) for k,v in d.items()}
    obs=obs[good]; fbg=fbg[good]; N=obs.shape[0]
    print('  profiles total',N0,'used',N,'dropped',int((~good).sum()),flush=True)
    atm,sfc,hyd,clear=build(d); am,sm,hm=scale(atm,sfc,hyd)
    print('  clear',int(clear.sum()),'cloudy',int((~clear).sum()),flush=True)
    xb=np.concatenate([d['T'],d['q']],1).astype(np.float64)
    sigma=np.concatenate([np.std(d['T'],axis=0),np.std(d['q'],axis=0)]).astype(np.float64)
    B=(sigma[:,None]*C)*sigma[None,:]; B=B+0.0025*np.diag(sigma**2); B=0.5*(B+B.T); Binv=np.linalg.pinv(B)
    R=np.diag(np.maximum(np.std(obs-fbg,axis=0),0.5)**2+0.3**2); Rinv=np.linalg.inv(R)
    K_list={'ARMS':np.concatenate([d['jac_T'],d['jac_Q']],axis=2).astype(np.float64)}
    models={}
    for name in ['transformer','mlp','bilstm']:
        m=load_model(name); print('  jacobian',name,flush=True)
        if name=='bilstm':
            try: K_list[name]=jacobian_fd(m,am,sm,hm,clear).astype(np.float64)
            except Exception as e: print('   fd failed %r'%e,flush=True); raise
        else:
            try: K_list[name]=jacobian_ad(m,am,sm,hm,clear).astype(np.float64)
            except Exception as e: print('   ad failed %r; fd fallback'%e,flush=True); K_list[name]=jacobian_fd(m,am,sm,hm,clear).astype(np.float64)
        models[name]=m
    results={}
    for name in NAMES:
        K=K_list[name]
        KtR=np.einsum('nij,jk->nik',K.transpose(0,2,1),Rinv)
        H=Binv[None]+np.einsum('nij,njk->nik',KtR,K)+1e-6*np.eye(74)[None]
        rhs=np.einsum('nij,nj->ni',KtR,obs-fbg)
        dx=np.linalg.solve(H,rhs[...,None])[...,0]
        x=xb+dx; x[:,:37]=np.clip(x[:,:37],180,330); x[:,37:]=np.clip(x[:,37:],0,30)
        if name=='ARMS':
            yhat=fbg+np.einsum('nij,nj->ni',K,dx)
        else:
            d2=dict(d); d2['T']=x[:,:37].astype(np.float32); d2['q']=x[:,37:].astype(np.float32)
            a2,s2,h2,c2=build(d2); am2,sm2,hm2=scale(a2,s2,h2)
            yhat=forward(models[name],am2,sm2,hm2,c2)
        S=np.linalg.inv(H); A=np.einsum('nij,njk->nik',S,np.einsum('nij,njk->nik',KtR,K)); dfs=float(np.mean(np.trace(A,axis1=1,axis2=2)))
        rb=rmse(obs,fbg); ra=rmse(obs,yhat); bias=float(np.mean(yhat-obs)); corr=float(np.corrcoef(obs.ravel(),yhat.ravel())[0,1])
        results[name]=dict(x=x,xb=xb,y_obs=obs,y_hat=yhat,dfs=dfs,rmse_ob=rb,rmse_oa=ra,bias_oa=bias,corr_oa=corr)
        rows.append([st,LABEL[name],rb,ra,bias,corr,dfs])
        print('  %-12s O-B %.3f  O-A %.3f  bias %.3f  corr %.3f  DFS %.2f'%(name,rb,ra,bias,corr,dfs),flush=True)
    np.savez_compressed(OUT/f'{st}_retrieval.npz',**{k:np.asarray(v) for k,v in results.items()})
    summary[st]=results
    print('  station done in %.1fs'%(time.time()-t0),flush=True)

with open(OUT/'station_retrieval_metrics.csv','w') as f:
    f.write('station,model,rmse_ob,rmse_oa,bias_oa,corr_oa,dfs\n')
    for r in rows: f.write(','.join(str(x) for x in r)+'\n')

print('ALL_DONE',flush=True)
