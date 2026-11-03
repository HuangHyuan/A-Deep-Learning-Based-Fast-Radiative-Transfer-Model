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

"""All-sky L37-input/L137-target Transformer context/gate ablations."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = _QJRMS_CORE
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
sys.path.insert(0, str(PROJECT))
from GB_model import UnifiedAllSkyRTM_Transformer

NLEV, NCHAN = 37, 22

class SceneOutputVariant(nn.Module):
    def __init__(self, mode, hidden_dim, n_layers, output_dim, context_dim):
        super().__init__()
        self.mode = mode
        self.n_layers = n_layers
        self.context_dim = context_dim
        self.fc_BT = nn.Linear(n_layers, output_dim)
        if mode in {'gate', 'fixed'}:
            self.fc_clear = nn.Linear(hidden_dim, 1)
            self.fc_cloudy = nn.Linear(hidden_dim, 1)
        else:
            self.fc_single = nn.Linear(hidden_dim, 1)
        if mode == 'gate':
            self.scene_gate = nn.Sequential(
                nn.Linear(context_dim, 32), nn.ReLU(), nn.Linear(32, 2), nn.Softmax(dim=-1)
            )

    def forward(self, x, cloud_context):
        if self.mode == 'single':
            mixed = self.fc_single(x).squeeze(-1)
        elif self.mode == 'fixed':
            mixed = 0.5 * (self.fc_clear(x).squeeze(-1) + self.fc_cloudy(x).squeeze(-1))
        else:
            if cloud_context is None:
                clear = self.fc_clear(x).squeeze(-1)
                weights = self.scene_gate(torch.zeros(x.shape[0], self.context_dim, device=x.device))
                mixed = weights[:, 0].unsqueeze(-1) * clear
            else:
                weights = self.scene_gate(cloud_context)
                mixed = (weights[:, 0].unsqueeze(-1) * self.fc_clear(x).squeeze(-1)
                         + weights[:, 1].unsqueeze(-1) * self.fc_cloudy(x).squeeze(-1))
        return self.fc_BT(mixed)


class VariantTransformer(UnifiedAllSkyRTM_Transformer):
    def __init__(self, variant, **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        if variant in {'fixed', 'single'}:
            self.scene_aware_output = SceneOutputVariant(
                variant, kwargs.get('hidden_dim', 160), kwargs.get('num_layers', 37),
                kwargs.get('num_channels', 22), kwargs.get('hydro_dim', 4),
            )

    def forward(self, x_atm, x_surf, x_hydro=None):
        batch_size = x_atm.size(0); device = x_atm.device
        x_var1 = x_atm[:, :, 0].unsqueeze(-1)
        x_var2 = x_atm[:, :, 1].unsqueeze(-1)
        x_const3 = x_atm[:, :, 2].unsqueeze(-1).detach()
        atm_encoded = self.atm_encoder(torch.cat([x_var1, x_var2, x_const3], dim=-1))
        sfc_var = x_surf[:, :3]; sfc_const = x_surf[:, 3:].detach()
        surf_encoded = self.surf_encoder(torch.cat([sfc_var, sfc_const], dim=-1)).unsqueeze(1).repeat(1, self.num_layers, 1)
        if x_hydro is None:
            hydro_encoded = torch.zeros(batch_size, self.num_layers, self.hydro_encoder[-1].out_features,
                                        device=device, dtype=x_atm.dtype)
            cloud_context = None
        else:
            constrained = self.adaptive_constraint(x_hydro)[0] if self.use_adaptive_constraint else x_hydro
            if constrained.shape[1] != self.num_layers:
                if constrained.shape[1] < self.num_layers:
                    constrained = F.pad(constrained, (0, 0, 0, self.num_layers - constrained.shape[1]))
                else:
                    constrained = constrained[:, :self.num_layers]
            hydro_encoded = self.hydro_encoder(constrained)
            cloud_context = constrained.sum(dim=1) if self.variant == 'sum' else constrained.mean(dim=1)
        combined = torch.cat([atm_encoded, surf_encoded, hydro_encoded], dim=-1)
        src = combined + self.pos_encoding
        for block in self.transformer:
            src = block(src)
        return self.scene_aware_output(src + combined, cloud_context)


class MatchedDataset(Dataset):
    def __init__(self, l37_path, l137_path, split, max_samples=None):
        arrays = []
        labels = []
        with h5py.File(l37_path, 'r') as a, h5py.File(l137_path, 'r') as b:
            for scene, label in (('clear', 0), ('cloudy', 1)):
                ga, gb = a[split][scene], b[split][scene]
                n = len(ga['atm'])
                atm = ga['atm'][:].astype(np.float32)
                sfc = ga['sfc'][:].astype(np.float32)
                out = gb['out'][:].astype(np.float32)
                hydro = ga['hydro'][:].astype(np.float32) if scene == 'cloudy' else np.zeros((n, NLEV - 1, 4), np.float32)
                arrays.append((atm, sfc, hydro, out))
                labels.append(np.full(n, label, np.int8))
        if max_samples is not None:
            half = max_samples // 2
            take = [min(half, len(arrays[i][0])) for i in range(2)]
            arrays = [tuple(v[:take[i]] for v in arrays[i]) for i in range(2)]
            labels = [labels[i][:take[i]] for i in range(2)]
        self.atm = np.concatenate([x[0] for x in arrays])
        self.sfc = np.concatenate([x[1] for x in arrays])
        self.hydro = np.concatenate([x[2] for x in arrays])
        self.out = np.concatenate([x[3] for x in arrays])
        self.label = np.concatenate(labels)
        self.n = len(self.atm)

    def __len__(self): return self.n

    def __getitem__(self, i):
        return self.atm[i], self.sfc[i], self.hydro[i], self.out[i], self.label[i]


def build_model(variant):
    common = dict(num_layers=NLEV, num_channels=NCHAN, atm_dim=3, surf_dim=5, hydro_dim=4,
                  hidden_dim=160, nhead=4, num_transformer_layers=3)
    return UnifiedAllSkyRTM_Transformer(**common) if variant == 'baseline' else VariantTransformer(variant, **common)


def forward_batch(model, atm, sfc, hydro, label):
    out = torch.empty((atm.shape[0], NCHAN), device=atm.device, dtype=atm.dtype)
    clear = label == 0; cloudy = ~clear
    if clear.any(): out[clear] = model(atm[clear], sfc[clear])
    if cloudy.any(): out[cloudy] = model(atm[cloudy], sfc[cloudy], hydro[cloudy])
    return out


def validation_loss(model, loader, device):
    model.eval(); total = 0.; count = 0
    with torch.no_grad():
        for atm, sfc, hydro, target, label in loader:
            atm, sfc, hydro, target, label = [x.to(device) for x in (atm, sfc, hydro, target, label)]
            pred = forward_batch(model, atm, sfc, hydro, label)
            total += F.mse_loss(pred, target, reduction='sum').item()
            count += target.numel()
    return total / count


def evaluate(model, loader, y_scaler, device):
    model.eval(); preds=[]; truth=[]
    with torch.no_grad():
        for atm,sfc,hydro,target,label in loader:
            atm,sfc,hydro,target,label=[x.to(device) for x in (atm,sfc,hydro,target,label)]
            preds.append(forward_batch(model,atm,sfc,hydro,label).cpu().numpy())
            truth.append(target.cpu().numpy())
    return y_scaler.inverse_transform(np.concatenate(preds)), y_scaler.inverse_transform(np.concatenate(truth))


def metric_rows(model_name, pred, truth, label):
    rows=[]
    for tag, mask in (('all',np.ones(len(truth),bool)),('clear',label==0),('cloudy',label==1)):
        p=pred[mask]; t=truth[mask]; e=p-t
        rows.append({'model':model_name,'scene':tag,'n':int(mask.sum()),
                     'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e))),
                     'bias':float(np.mean(e)),
                     'corr':float(np.corrcoef(t.ravel(),p.ravel())[0,1]) if np.std(t)>0 and np.std(p)>0 else np.nan})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--max-samples',type=int,default=None)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--variants',default='baseline,sum,fixed,single')
    ap.add_argument('--out',type=Path,default=Path(os.environ.get('QJRMS_RUNS_DIR', _QJRMS_ROOT / 'runs')) / 'scene_gate_ablation')
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.out.mkdir(parents=True,exist_ok=True)
    data37=Path(os.environ.get('QJRMS_DATA_L37', _QJRMS_ROOT / 'data' / 'bt_L37.h5'))
    data137=Path(os.environ.get('QJRMS_DATA_L137', _QJRMS_ROOT / 'data' / 'bt_L137.h5'))
    y_scaler=joblib.load(data137.parent/'scaler_info.pkl')['y']
    train_ds=MatchedDataset(data37,data137,'train',args.max_samples)
    valid_ds=MatchedDataset(data37,data137,'valid',args.max_samples)
    test_ds=MatchedDataset(data37,data137,'test',args.max_samples)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    valid_loader=DataLoader(valid_ds,batch_size=args.batch_size*2,shuffle=False,num_workers=4,pin_memory=True)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size*2,shuffle=False,num_workers=4,pin_memory=True)
    print('sizes',len(train_ds),len(valid_ds),len(test_ds),flush=True)
    all_metrics=[]
    for variant in [x.strip() for x in args.variants.split(',') if x.strip()]:
        print('\n===',variant,'===',flush=True)
        model=build_model(variant).to(device)
        params=sum(p.numel() for p in model.parameters())
        print('parameters',params,flush=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,min_lr=1e-6)
        best=float('inf'); bad=0; ckpt=args.out/f'{variant}.pth'
        for epoch in range(1,args.epochs+1):
            model.train(); total=0.; count=0; t0=time.time()
            for atm,sfc,hydro,target,label in train_loader:
                atm,sfc,hydro,target,label=[x.to(device) for x in (atm,sfc,hydro,target,label)]
                optimizer.zero_grad(set_to_none=True)
                pred=forward_batch(model,atm,sfc,hydro,label)
                loss=F.mse_loss(pred,target)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                total += loss.item()*target.numel(); count += target.numel()
            train_loss=total/count
            val_loss=validation_loss(model,valid_loader,device)
            scheduler.step(val_loss)
            print(f'epoch {epoch}/{args.epochs} train={train_loss:.6e} val={val_loss:.6e} time={time.time()-t0:.1f}s',flush=True)
            if val_loss<best-1e-12:
                best=val_loss; bad=0
                torch.save({'variant':variant,'parameters':params,'state_dict':model.state_dict(),
                            'best_val_loss':best,'epoch':epoch},ckpt)
            else:
                bad+=1
                if bad>=8:
                    print('early stop',flush=True); break
        checkpoint=torch.load(ckpt,map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['state_dict']); model.eval()
        pred,truth=evaluate(model,test_loader,y_scaler,device)
        all_metrics.extend(metric_rows(variant,pred,truth,test_ds.label))
        np.savez_compressed(args.out/f'{variant}_test_predictions.npz',pred=pred,truth=truth,label=test_ds.label)
        del model,optimizer
        torch.cuda.empty_cache()
    with (args.out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_metrics[0])); w.writeheader(); w.writerows(all_metrics)
    (args.out/'metadata.json').write_text(json.dumps({'train_samples':len(train_ds),'valid_samples':len(valid_ds),
        'test_samples':len(test_ds),'epochs':args.epochs,'batch_size':args.batch_size,'device':str(device),
        'input_data':str(data37),'target_data':str(data137)},indent=2),encoding='utf-8')
    print('saved',args.out,flush=True)

if __name__=='__main__':
    main()
