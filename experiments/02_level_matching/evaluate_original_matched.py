from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Evaluate the four original L37 checkpoints on the matched revision test set.

The checkpoints consume 37-level atmospheric and hydrometeor fields. Their
surface TCLW is the prepared TCLW column stored in bt_L37, which is derived
from the L137-level cloud-water profile and is compatible with the original
training scaler. The raw L137 TCLW product has a different magnitude and is
retained only as a diagnostic option.
"""

import argparse
import os
import csv
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np
import torch

PROJECT_ROOT = _QJRMS_CORE
ORIGINAL_ROOT = Path(os.environ.get('QJRMS_MODEL_DIR', _QJRMS_ROOT / 'model_assets'))
sys.path.insert(0, str(PROJECT_ROOT))

from GB_model import (  # noqa: E402
    UnifiedAllSkyRTM_BiLSTM,
    UnifiedAllSkyRTM_MLP,
    UnifiedAllSkyRTM_Transformer,
)


MODEL_SPECS = {
    'Unified-Trans': {
        'kind': 'unified',
        'checkpoint': ORIGINAL_ROOT / 'best_unified_model_ATT.pth',
        'factory': lambda: UnifiedAllSkyRTM_Transformer(
            hidden_dim=160, nhead=4, num_transformer_layers=3
        ),
    },
    'Unified-BiLSTM': {
        'kind': 'unified',
        'checkpoint': ORIGINAL_ROOT / 'best_unified_model_BiLSTM.pth',
        'factory': lambda: UnifiedAllSkyRTM_BiLSTM(
            hidden_dim=180, num_bi_lstm_blocks=2
        ),
    },
    'Unified-MLP': {
        'kind': 'unified',
        'checkpoint': ORIGINAL_ROOT / 'best_unified_model_MLP.pth',
        'factory': lambda: UnifiedAllSkyRTM_MLP(
            hidden_dim=12, num_mlp_blocks=3
        ),
    },
    'Scene-Specific': {
        'kind': 'scene',
        'checkpoints': {
            'clear': ORIGINAL_ROOT / 'best_clear_model.pth',
            'cloudy': ORIGINAL_ROOT / 'best_cloudy_model.pth',
        },
        'factory': lambda: UnifiedAllSkyRTM_Transformer(
            hidden_dim=160, nhead=2, num_transformer_layers=3
        ),
    },
}


def inverse_vertical(scaled, scalers):
    result = np.empty_like(scaled, dtype=np.float32)
    for level, scaler in enumerate(scalers):
        result[:, level] = scaler.inverse_transform(scaled[:, level])
    return result


def transform_vertical(raw, scalers):
    result = np.empty_like(raw, dtype=np.float32)
    for level, scaler in enumerate(scalers):
        result[:, level] = scaler.transform(raw[:, level])
    return result


def load_raw_test(data_path, limit):
    scalers = joblib.load(data_path.parent / 'scaler_info.pkl')
    out = {}
    with h5py.File(data_path, 'r') as h5:
        for scene in ('clear', 'cloudy'):
            group = h5['test'][scene]
            n = len(group['atm'])
            if limit is not None:
                n = min(n, limit)
            raw = {
                'atm': inverse_vertical(group['atm'][:n], scalers['atm']),
                'sfc': scalers['sfc'].inverse_transform(group['sfc'][:n]),
                'out': scalers['y'].inverse_transform(group['out'][:n]),
            }
            if scene == 'cloudy':
                raw['hydro'] = inverse_vertical(group['hydro'][:n], scalers['hydro'])
            out[scene] = raw
    return out


def apply_tclw_source(raw, mode, source_root, date, limit):
    # l37-sfc is the training-consistent field. l137 is diagnostic only.
    if mode == 'l37-sfc':
        return
    source = source_root / f'ifs_cloudy_{date}.nc'
    with h5py.File(source, 'r') as h5:
        tclw = np.asarray(h5['TCLW'][:], dtype=np.float32).reshape(-1)
    n = len(raw['cloudy']['sfc'])
    if len(tclw) < n:
        raise ValueError(f'L137 source has too few profiles: {source}')
    raw['cloudy']['sfc'][:, 3] = tclw[:n]


def prepare_original_inputs(raw):
    atm_scalers = joblib.load(ORIGINAL_ROOT / 'scaler_atm_global.pkl')
    sfc_scaler = joblib.load(ORIGINAL_ROOT / 'scaler_sfc_global.pkl')
    hydro_scalers = joblib.load(ORIGINAL_ROOT / 'scaler_hydro_cloudy.pkl')
    prepared = {}
    for scene in ('clear', 'cloudy'):
        item = {
            'atm': transform_vertical(raw[scene]['atm'], atm_scalers),
            'sfc': sfc_scaler.transform(raw[scene]['sfc']).astype(np.float32),
            'out': raw[scene]['out'],
        }
        if scene == 'cloudy':
            item['hydro'] = transform_vertical(raw[scene]['hydro'], hydro_scalers)
        prepared[scene] = item
    return prepared


def predict(model, scene, data, device, batch_size):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data['atm']), batch_size):
            end = min(start + batch_size, len(data['atm']))
            atm = torch.from_numpy(data['atm'][start:end]).to(device)
            sfc = torch.from_numpy(data['sfc'][start:end]).to(device)
            if scene == 'cloudy':
                hydro = torch.from_numpy(data['hydro'][start:end]).to(device)
                pred = model(atm, sfc, hydro)
            else:
                pred = model(atm, sfc)
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def metric_rows(model_name, scene, truth, pred):
    error = pred.astype(np.float64) - truth.astype(np.float64)
    rows = []
    for channel in range(truth.shape[1]):
        e = error[:, channel]
        t = truth[:, channel].astype(np.float64)
        p = pred[:, channel].astype(np.float64)
        corr = np.corrcoef(t, p)[0, 1] if np.std(t) > 1e-12 and np.std(p) > 1e-12 else np.nan
        rows.append({
            'model': model_name,
            'scene': scene,
            'channel': channel,
            'n': int(len(e)),
            'rmse': float(np.sqrt(np.mean(e * e))),
            'bias': float(np.mean(e)),
            'mae': float(np.mean(np.abs(e))),
            'corr': float(corr),
        })
    all_error = error.ravel()
    rows.append({
        'model': model_name,
        'scene': scene,
        'channel': 'all',
        'n': int(all_error.size),
        'rmse': float(np.sqrt(np.mean(all_error * all_error))),
        'bias': float(np.mean(all_error)),
        'mae': float(np.mean(np.abs(all_error))),
        'corr': np.nan,
    })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data', type=Path,
        default=Path(os.environ.get('QJRMS_DATA_L37', _QJRMS_ROOT / 'data' / 'bt_L37.h5')),
    )
    parser.add_argument(
        '--l137-source', type=Path,
        default=Path(os.environ.get('QJRMS_L137_SOURCE', _QJRMS_ROOT / 'data' / 'L137')),
    )
    parser.add_argument('--date', default='2023-12-16')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--tclw-source', choices=['l37-sfc', 'l137'], default='l37-sfc')
    parser.add_argument(
        '--out', type=Path,
        default=Path(os.environ.get('QJRMS_RUNS_DIR', _QJRMS_ROOT / 'runs')) / 'model_compare_original_matched.csv',
    )
    parser.add_argument('--predictions', type=Path, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw = load_raw_test(args.data, args.limit)
    apply_tclw_source(raw, args.tclw_source, args.l137_source, args.date, args.limit)
    data = prepare_original_inputs(raw)
    y_scaler = joblib.load(ORIGINAL_ROOT / 'scaler_y.pkl')

    print('device:', device)
    for scene in ('clear', 'cloudy'):
        print(scene, {key: value.shape for key, value in data[scene].items()})
        print(
            scene,
            'scaled atm range',
            float(data[scene]['atm'].min()),
            float(data[scene]['atm'].max()),
            'sfc range',
            float(data[scene]['sfc'].min()),
            float(data[scene]['sfc'].max()),
        )

    all_rows = []
    predictions = {}
    for model_name, spec in MODEL_SPECS.items():
        for scene in ('clear', 'cloudy'):
            checkpoint = (
                spec['checkpoints'][scene]
                if spec['kind'] == 'scene'
                else spec['checkpoint']
            )
            model = spec['factory']().to(device).eval()
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state)
            pred = predict(model, scene, data[scene], device, args.batch_size)
            pred = y_scaler.inverse_transform(pred)
            truth = data[scene]['out']
            rows = metric_rows(model_name, scene, truth, pred)
            all_rows.extend(rows)
            predictions[f'{model_name}__{scene}'] = pred.astype(np.float32)
            overall = rows[-1]
            print(
                f"{model_name:18s} {scene:6s} "
                f"RMSE={overall['rmse']:.6f} K bias={overall['bias']:+.6f} K"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.out, all_rows)
    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.predictions, **predictions)
    print('saved', args.out)


if __name__ == '__main__':
    main()
