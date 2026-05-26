import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, jacrev
import numpy as np
from typing import Optional
import h5py
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import math
import joblib
from GB_model import UnifiedAllSkyRTM_Transformer

# ============================================================================
# 1. Dataset Class
# ============================================================================

class JacobianDataset(Dataset):
    """PyTorch Dataset that provides input/output pairs and ground-truth Jacobians."""
    def __init__(self, h5_path, mode='train', data_type='both', max_samples=None):
        """
        Args:
            h5_path: Path to HDF5 file containing data
            mode: 'train' or 'valid'
            data_type: 'clear', 'cloudy', or 'both'
            max_samples: Maximum number of samples to use (optional)
        """
        self.h5_path = h5_path
        self.mode = mode
        self.data_type = data_type
        
        with h5py.File(h5_path, 'r') as h5_file:
            if mode == 'train':
                data_group = h5_file['train']
            else:
                data_group = h5_file['valid']
            
            # Load Jacobian statistics (min/max for normalization)
            stats_group = h5_file['jacobian_stats']
            self.jacobian_stats = {
                'atm_min': torch.tensor(stats_group['atm_min'][:], dtype=torch.float32),
                'atm_max': torch.tensor(stats_group['atm_max'][:], dtype=torch.float32),
                'sfc_min': torch.tensor(stats_group['sfc_min'][:], dtype=torch.float32),
                'sfc_max': torch.tensor(stats_group['sfc_max'][:], dtype=torch.float32),
                'cloud_min': torch.tensor(stats_group['cloud_min'][:], dtype=torch.float32),
                'cloud_max': torch.tensor(stats_group['cloud_max'][:], dtype=torch.float32)
            }
            
            # Load data according to data_type
            if data_type == 'clear' or data_type == 'both':
                clear_group = data_group['clear']
                self.clear_atm = clear_group['atm'][:]
                self.clear_sfc = clear_group['sfc'][:]
                self.clear_out = clear_group['out'][:]
                self.clear_jac_atm = clear_group['jac_atm'][:]
                self.clear_jac_sfc = clear_group['jac_sfc'][:]
                self.clear_len = len(self.clear_atm)
            else:
                self.clear_len = 0
            
            if data_type == 'cloudy' or data_type == 'both':
                cloudy_group = data_group['cloudy']
                self.cloudy_atm = cloudy_group['atm'][:]
                self.cloudy_sfc = cloudy_group['sfc'][:]
                self.cloudy_hydro = cloudy_group['hydro'][:]
                self.cloudy_out = cloudy_group['out'][:]
                self.cloudy_jac_atm = cloudy_group['jac_atm'][:]
                self.cloudy_jac_sfc = cloudy_group['jac_sfc'][:]
                self.cloudy_jac_cloud = cloudy_group['jac_cloud'][:]
                self.cloudy_len = len(self.cloudy_atm)
            else:
                self.cloudy_len = 0
            
            self.total_len = self.clear_len + self.cloudy_len
            
            # Optionally subsample
            if max_samples and max_samples < self.total_len:
                indices = np.random.choice(self.total_len, max_samples, replace=False)
                self._subset_data(indices)
    
    def _subset_data(self, indices):
        """Create a subset of the data given indices."""
        # Split indices into clear and cloudy parts
        clear_indices = indices[indices < self.clear_len]
        cloudy_indices = indices[indices >= self.clear_len] - self.clear_len
        
        if len(clear_indices) > 0:
            self.clear_atm = self.clear_atm[clear_indices]
            self.clear_sfc = self.clear_sfc[clear_indices]
            self.clear_out = self.clear_out[clear_indices]
            self.clear_jac_atm = self.clear_jac_atm[clear_indices]
            self.clear_jac_sfc = self.clear_jac_sfc[clear_indices]
            self.clear_len = len(clear_indices)
        
        if len(cloudy_indices) > 0:
            self.cloudy_atm = self.cloudy_atm[cloudy_indices]
            self.cloudy_sfc = self.cloudy_sfc[cloudy_indices]
            self.cloudy_hydro = self.cloudy_hydro[cloudy_indices]
            self.cloudy_out = self.cloudy_out[cloudy_indices]
            self.cloudy_jac_atm = self.cloudy_jac_atm[cloudy_indices]
            self.cloudy_jac_sfc = self.cloudy_jac_sfc[cloudy_indices]
            self.cloudy_jac_cloud = self.cloudy_jac_cloud[cloudy_indices]
            self.cloudy_len = len(cloudy_indices)
        
        self.total_len = self.clear_len + self.cloudy_len
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.clear_len:
            return {
                'atm_input': torch.tensor(self.clear_atm[idx], dtype=torch.float32),
                'sfc_input': torch.tensor(self.clear_sfc[idx], dtype=torch.float32),
                'hydro_input': torch.zeros((36, 4), dtype=torch.float32),  # dummy zeros for clear-sky
                'output': torch.tensor(self.clear_out[idx], dtype=torch.float32),
                'jac_atm': torch.tensor(self.clear_jac_atm[idx], dtype=torch.float32),
                'jac_sfc': torch.tensor(self.clear_jac_sfc[idx], dtype=torch.float32),
                'jac_cloud': torch.zeros((22, 36, 4), dtype=torch.float32),  # dummy zeros for clear-sky
                'is_cloudy': torch.tensor(0.0, dtype=torch.float32)
            }
        else:
            cloudy_idx = idx - self.clear_len
            return {
                'atm_input': torch.tensor(self.cloudy_atm[cloudy_idx], dtype=torch.float32),
                'sfc_input': torch.tensor(self.cloudy_sfc[cloudy_idx], dtype=torch.float32),
                'hydro_input': torch.tensor(self.cloudy_hydro[cloudy_idx], dtype=torch.float32),
                'output': torch.tensor(self.cloudy_out[cloudy_idx], dtype=torch.float32),
                'jac_atm': torch.tensor(self.cloudy_jac_atm[cloudy_idx], dtype=torch.float32),
                'jac_sfc': torch.tensor(self.cloudy_jac_sfc[cloudy_idx], dtype=torch.float32),
                'jac_cloud': torch.tensor(self.cloudy_jac_cloud[cloudy_idx], dtype=torch.float32),
                'is_cloudy': torch.tensor(1.0, dtype=torch.float32)
            }
    
    def get_jacobian_stats(self):
        """Return the Jacobian min/max statistics for normalization."""
        return self.jacobian_stats

# ============================================================================
# 2. Core Functions: Jacobian Computation and Normalization
# ============================================================================

def reverse_jacobian_sfc(jac_matrix, sfc_scaler_info, y_scaler_info):
    """
    Reverse Min-Max scaling for surface Jacobians.

    Scaling assumptions:
        y_scaled = (y - y_min) / (y_max - y_min)
        x_scaled = (x - x_min) / (x_max - x_min)

    Then physical Jacobian is:
        ∂y/∂x = (∂y_scaled/∂x_scaled) * ((y_max - y_min) / (x_max - x_min))

    Args:
        jac_matrix: (B, 22, n_params) - scaled Jacobian (n_params typically 3 or 4)
        sfc_scaler_info: dict with keys 'min', 'max' of shape (n_params,)
        y_scaler_info: dict with keys 'min', 'max' of shape (22,)

    Returns:
        Inverse-scaled Jacobian (B, 22, n_params)
    """
    B, channels, n_params = jac_matrix.shape
    
    # Convert to tensor if needed
    if isinstance(jac_matrix, np.ndarray):
        jac_matrix = torch.from_numpy(jac_matrix).float()
    
    device = jac_matrix.device

    # Output y range
    y_max = torch.as_tensor(y_scaler_info['max'], dtype=torch.float32, device=device)   # (22,)
    y_min = torch.as_tensor(y_scaler_info['min'], dtype=torch.float32, device=device)   # (22,)
    y_range = y_max - y_min  # (22,)

    # Surface input range
    sfc_max = torch.as_tensor(sfc_scaler_info['max'], dtype=torch.float32, device=device)  # (n_params,)
    sfc_min = torch.as_tensor(sfc_scaler_info['min'], dtype=torch.float32, device=device)  # (n_params,)
    sfc_range = sfc_max - sfc_min  # (n_params,)

    # Add dimensions for broadcasting
    y_range = y_range.view(1, -1, 1)      # (1, 22, 1)
    sfc_range = sfc_range.view(1, 1, -1)  # (1, 1, n_params)

    # Avoid division by zero: if range is zero, set to 1 temporarily
    sfc_range_safe = torch.where(
        sfc_range == 0,
        torch.ones_like(sfc_range),
        sfc_range
    )

    # Apply inverse scaling
    jac_reversed = jac_matrix * (y_range / sfc_range_safe)

    # If input range is zero (max==min), the derivative should be zero
    jac_reversed = torch.where(
        sfc_range == 0,
        torch.zeros_like(jac_reversed),
        jac_reversed
    )

    return jac_reversed

def reverse_jacobian_batch(jac_matrix, atm_scaler_info, y_scaler_info):
    """
    Reverse Min-Max scaling for atmospheric/hydrometeor Jacobians (4D tensors).

    Args:
        jac_matrix: (B, 22, seq_len, n_params) - scaled Jacobian
        atm_scaler_info: dict with keys 'max_array', 'min_array' of shape (seq_len, n_params)
        y_scaler_info: dict with keys 'max', 'min' of shape (22,)

    Returns:
        Inverse-scaled Jacobian (B, 22, seq_len, n_params)
    """
    B, channels, seq_len, n_params = jac_matrix.shape
    
    if isinstance(jac_matrix, np.ndarray):
        jac_matrix = torch.from_numpy(jac_matrix).float()
    
    device = jac_matrix.device

    # Output range
    y_max = torch.as_tensor(y_scaler_info['max'], dtype=torch.float32, device=device)   # (22,)
    y_min = torch.as_tensor(y_scaler_info['min'], dtype=torch.float32, device=device)   # (22,)
    y_range = y_max - y_min  # (22,)
    
    # Input (atm/hydro) range per level and variable
    atm_max_array = torch.as_tensor(atm_scaler_info['max_array'], dtype=torch.float32, device=device)  # (seq_len, n_params)
    atm_min_array = torch.as_tensor(atm_scaler_info['min_array'], dtype=torch.float32, device=device)  # (seq_len, n_params)
    atm_range = atm_max_array - atm_min_array  # (seq_len, n_params)

    # Add dimensions for broadcasting
    y_range = y_range.view(1, -1, 1, 1)          # (1, 22, 1, 1)
    atm_range = atm_range.view(1, 1, seq_len, n_params)  # (1, 1, seq_len, n_params)

    # Avoid division by zero
    atm_range_safe = torch.where(
        atm_range == 0,
        torch.ones_like(atm_range),
        atm_range
    )

    jac_reversed = jac_matrix * (y_range / atm_range_safe)

    # Zero out where input range is zero
    jac_reversed = torch.where(
        atm_range == 0,
        torch.zeros_like(jac_reversed),
        jac_reversed
    )

    return jac_reversed

def compute_model_jacobian(model, atm_input, sfc_input, hydro_input=None, 
                          batch_size=8, device='cuda'):
    """
    Compute Jacobians of the model output with respect to inputs.

    Args:
        model: UnifiedAllSkyRTM_Transformer model
        atm_input: (B, 37, 3) atmospheric profiles
        sfc_input: (B, 4) surface variables
        hydro_input: (B, 36, 3) hydrometeor profiles (optional)
        batch_size: Batch size for Jacobian computation (memory control)
        device: Device to use

    Returns:
        J_atm: (B, 22, 37, 3) Jacobian w.r.t. atm_input
        J_sfc: (B, 22, 4) Jacobian w.r.t. sfc_input
        J_hydro: (B, 22, 36, 3) Jacobian w.r.t. hydro_input (zeros if clear-sky)
    """
    model.eval()
    B = atm_input.shape[0]
    n_channels = 22
    n_atm_levels = 37
    n_atm = 3
    n_sfc = 5
    n_hydro = 4
    n_hydro_levels = 36
    
    # Pre-allocate results
    J_atm = torch.zeros((B, n_channels, n_atm_levels, n_atm), 
                        dtype=torch.float32, device=device)
    J_sfc = torch.zeros((B, n_channels, n_sfc), 
                       dtype=torch.float32, device=device)
    
    if hydro_input is not None:
        J_hydro = torch.zeros((B, n_channels, n_hydro_levels, n_hydro), 
                             dtype=torch.float32, device=device)
    else:
        J_hydro = torch.zeros((B, n_channels, n_hydro_levels, n_hydro), 
                             dtype=torch.float32, device=device)
    
    # Single-sample forward functions (for vmap)
    def forward_clear(atm, sfc):
        return model(atm.unsqueeze(0), sfc.unsqueeze(0)).squeeze(0)
    
    def forward_cloudy(atm, sfc, hydro):
        return model(atm.unsqueeze(0), sfc.unsqueeze(0), hydro.unsqueeze(0)).squeeze(0)
    
    # Process in batches to avoid OOM
    for i in range(0, B, batch_size):
        end = min(i + batch_size, B)
        batch_atm = atm_input[i:end].to(device)
        batch_sfc = sfc_input[i:end].to(device)
        
        if hydro_input is not None:
            batch_hydro = hydro_input[i:end].to(device)
            
            # Vectorized Jacobian using vmap and jacrev
            jac_fn = vmap(jacrev(forward_cloudy, argnums=(0, 1, 2)), 
                         in_dims=(0, 0, 0))
            J_atm_batch, J_sfc_batch, J_hydro_batch = jac_fn(
                batch_atm, batch_sfc, batch_hydro)
            
            J_atm[i:end] = J_atm_batch
            J_sfc[i:end] = J_sfc_batch
            J_hydro[i:end] = J_hydro_batch
        else:
            # Clear-sky case
            jac_fn = vmap(jacrev(forward_clear, argnums=(0, 1)), 
                         in_dims=(0, 0))
            J_atm_batch, J_sfc_batch = jac_fn(batch_atm, batch_sfc)
            
            J_atm[i:end] = J_atm_batch
            J_sfc[i:end] = J_sfc_batch
    
    return J_atm, J_sfc, J_hydro

# ============================================================================
# 3. Brightness Temperature Transformation and Jacobian Normalization
# ============================================================================

def log_transform(jac, eps=1e-8):
    """
    Safe log transform: sign(x) * log(|x| + eps).
    Used for stabilizing wide-range Jacobian values.
    """
    sign = torch.sign(jac)
    return sign * torch.log(torch.abs(jac) + eps)

def safe_normalize_jacobian(jac, min_vals, max_vals, epsilon=1e-8, 
                          use_log_transform=False, log_indices=None):
    """
    Stable Jacobian normalization supporting optional log transform for specific dimensions.

    Args:
        jac: Jacobian tensor of shape [..., D] (D is number of variables)
        min_vals: Minimum values, shape [D] or broadcastable to jac
        max_vals: Maximum values, shape [D] or broadcastable
        epsilon: Small constant for numerical stability
        use_log_transform: Whether to apply log transform before normalization
        log_indices: List of indices (last dimension) to apply log transform

    Returns:
        Normalized Jacobian tensor (values roughly in [0,1] or scaled appropriately)
    """
    # 1. Clean up inf/nan
    jac = torch.nan_to_num(jac, nan=0.0, posinf=1e4, neginf=-1e4)
    min_vals = torch.nan_to_num(min_vals, nan=0.0)
    max_vals = torch.nan_to_num(max_vals, nan=1.0)
    
    # 2. Expand dimensions if min_vals / max_vals are 1D
    if min_vals.dim() == 1:
        for _ in range(jac.dim() - min_vals.dim()):
            min_vals = min_vals.unsqueeze(0)
            max_vals = max_vals.unsqueeze(0)

    # Clip extreme values
    jac = torch.clamp(jac, min_vals, max_vals)

    # 3. Conditional log transform
    if use_log_transform and log_indices is not None and len(log_indices) > 0:
        # Transform bounds
        transformed_min = min_vals.clone()
        transformed_min[..., log_indices] = log_transform(transformed_min[..., log_indices])
        
        transformed_max = max_vals.clone()
        transformed_max[..., log_indices] = log_transform(transformed_max[..., log_indices])
        
        # Transform Jacobian itself
        transformed_jac = jac.clone()
        transformed_jac[..., log_indices] = log_transform(transformed_jac[..., log_indices])

        # Clip transformed values to the transformed bounds
        transformed_jac[..., log_indices] = torch.clamp(
            transformed_jac[..., log_indices], 
            transformed_min[..., log_indices], 
            transformed_max[..., log_indices]
        )
    else:
        transformed_jac = jac
        transformed_min = min_vals
        transformed_max = max_vals

    # 4. Min-Max normalization with fallback cases
    range_vals = transformed_max - transformed_min
    abs_max = torch.abs(transformed_max)
    
    # Masks for different scenarios
    wide_range = range_vals > epsilon                     # case 1: sufficient range
    small_range_nonzero_max = (~wide_range) & (abs_max > epsilon)  # case 2: small range but non-zero max
    zero_max = (~wide_range) & (abs_max <= epsilon)       # case 3: max ~ 0

    normalized = torch.zeros_like(transformed_jac)

    # Case 1: standard Min-Max scaling
    if wide_range.any():
        range_safe = torch.where(wide_range, range_vals, torch.ones_like(range_vals))
        normalized = torch.where(
            wide_range,
            (transformed_jac - transformed_min) / range_safe,
            normalized
        )

    # Case 2: small range but non-zero max -> scale by max (preserves sign)
    if small_range_nonzero_max.any():
        denom = torch.where(small_range_nonzero_max, transformed_max, torch.ones_like(transformed_max))
        normalized = torch.where(
            small_range_nonzero_max,
            transformed_jac / denom,
            normalized
        )

    # Case 3: max ~ 0 -> output zero (or keep original, here we keep zero)
    if zero_max.any():
        normalized = torch.where(zero_max, transformed_jac, normalized)

    # Optional global clipping to avoid extreme values
    normalized = torch.clamp(normalized, -10.0, 10.0)
    
    return normalized

# ============================================================================
# 4. Jacobian-Constrained Training Function
# ============================================================================

def train_with_jacobian_constraint(
    model, 
    h5_path, 
    save_path,
    scaler_info_path='./scalers/scaler_info.pkl',
    pretrained_model_path=None,
    lambda_jac=5e-4,
    jac_freq=1,
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    device='cuda',
):
    """
    Train a radiative transfer model with additional Jacobian consistency loss.

    The loss is: L_total = L_task + lambda_jac * L_jac,
    where L_jac compares the model's analytical Jacobian (via autodiff) against
    precomputed ground-truth Jacobians from a reference RTM.

    Args:
        model: PyTorch model to train
        h5_path: Path to HDF5 dataset containing Jacobian targets
        save_path: Where to save the best model checkpoint
        scaler_info_path: Path to pickled scaler info (min/max for inputs/outputs)
        pretrained_model_path: Optional path to load pretrained weights
        lambda_jac: Weight for Jacobian loss
        jac_freq: Frequency (in batches) to compute Jacobian loss
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
    """
    print("=" * 80)
    print("Starting Jacobian-constrained fine-tuning (full version)")
    print("=" * 80)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scaler information (min/max for inputs and outputs)
    print("\nLoading scaler information...")
    scaler_info = joblib.load(scaler_info_path)
    
    # Convert scaler info to tensors and move to device
    scaler_tensors = {}
    for key in scaler_info:
        if isinstance(scaler_info[key], dict):
            scaler_tensors[key] = {}
            for subkey in scaler_info[key]:
                if isinstance(scaler_info[key][subkey], np.ndarray):
                    scaler_tensors[key][subkey] = torch.from_numpy(
                        scaler_info[key][subkey]
                    ).float().to(device)
                else:
                    scaler_tensors[key][subkey] = scaler_info[key][subkey]
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = JacobianDataset(h5_path, mode='train', data_type='both')
    valid_dataset = JacobianDataset(h5_path, mode='valid', data_type='both')
    
    # Get Jacobian statistics (min/max) for normalization
    jacobian_stats = train_dataset.get_jacobian_stats()
    for key, val in jacobian_stats.items():
        jacobian_stats[key] = val.to(device)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size*2, shuffle=False, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Load pretrained weights if provided
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model: {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print("Pretrained model loaded successfully")
    
    model = model.to(device)
    
    # Print model parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Loss functions
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    
    # Training tracking
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    jac_losses = []
    task_losses = []
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ---------- Training Phase ----------
        model.train()
        epoch_train_loss = 0.0
        epoch_task_loss = 0.0
        epoch_jac_loss = 0.0
        train_steps = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                atm_input = batch['atm_input'].to(device)
                sfc_input = batch['sfc_input'].to(device)
                hydro_input = batch['hydro_input'].to(device)
                output_gt = batch['output'].to(device)
                jac_atm_gt = batch['jac_atm'].to(device)      # ground truth physical Jacobian (atm)
                jac_sfc_gt = batch['jac_sfc'].to(device)      # ground truth physical Jacobian (sfc)
                jac_cloud_gt = batch['jac_cloud'].to(device)  # ground truth physical Jacobian (cloud)
                is_cloudy = batch['is_cloudy'].to(device)
                
                optimizer.zero_grad()
                
                # Separate clear and cloudy samples
                clear_mask = (is_cloudy == 0)
                cloudy_mask = (is_cloudy == 1)

                n_clear = clear_mask.sum().item()
                n_cloudy = cloudy_mask.sum().item()
                n_total = atm_input.size(0)

                # Forward pass for radiance prediction
                radiance_pred = torch.zeros_like(output_gt)
                
                if clear_mask.any():
                    radiance_pred[clear_mask] = model(
                        atm_input[clear_mask], 
                        sfc_input[clear_mask]
                    )
                
                if cloudy_mask.any():
                    radiance_pred[cloudy_mask] = model(
                        atm_input[cloudy_mask], 
                        sfc_input[cloudy_mask], 
                        hydro_input[cloudy_mask]
                    )
                
                # Task loss (radiance prediction)
                task_loss = criterion_mse(radiance_pred, output_gt)
                
                # Jacobian constraint loss (computed optionally every jac_freq batches)
                atm_jac_loss = torch.tensor(0.0, device=device)
                sfc_jac_loss = torch.tensor(0.0, device=device)
                hydro_jac_loss = torch.tensor(0.0, device=device)

                if batch_idx % jac_freq == 0:
                    # ========== Compute model Jacobians via autodiff ==========
                    with torch.enable_grad():
                        # Clear-sky samples
                        if clear_mask.any() and clear_mask.sum() > 0:
                            atm_clear = atm_input[clear_mask].requires_grad_(True)
                            sfc_clear = sfc_input[clear_mask].requires_grad_(True)
                            
                            # Model Jacobians in scaled space
                            jac_atm_model_norm, jac_sfc_model_norm, _ = compute_model_jacobian(
                                model, atm_clear, sfc_clear, None, 
                                batch_size=batch_size, device=device
                            )

                            # ========== Inverse scaling to physical space ==========
                            # Atmospheric Jacobian
                            jac_atm_model_raw = reverse_jacobian_batch(
                                jac_atm_model_norm,
                                scaler_tensors['atm'],
                                scaler_tensors['y'],
                            )
                            # Surface Jacobian
                            jac_sfc_model_raw = reverse_jacobian_sfc(
                                jac_sfc_model_norm,
                                scaler_tensors['sfc'],
                                scaler_tensors['y']
                            )

                            # Ground truth physical Jacobians for clear-sky
                            jac_atm_gt_clear = jac_atm_gt[clear_mask]
                            jac_sfc_gt_clear = jac_sfc_gt[clear_mask]
                            
                            # ========== Normalize both model and GT Jacobians using dataset statistics ==========
                            jac_atm_model_norm_tb = safe_normalize_jacobian(
                                jac_atm_model_raw[:, :, :, :2],  # (B,22,37,2) - only temperature and water vapor
                                jacobian_stats['atm_min'],
                                jacobian_stats['atm_max'] 
                            )
                            jac_sfc_model_norm_tb = safe_normalize_jacobian(
                                jac_sfc_model_raw[:, :, :3],  # (B,22,3) - surface variables (skip cloud related)
                                jacobian_stats['sfc_min'],
                                jacobian_stats['sfc_max']
                            )
                            jac_atm_gt_norm_tb = safe_normalize_jacobian(
                                jac_atm_gt_clear,
                                jacobian_stats['atm_min'],
                                jacobian_stats['atm_max']
                            )
                            jac_sfc_gt_norm_tb = safe_normalize_jacobian(
                                jac_sfc_gt_clear,
                                jacobian_stats['sfc_min'],
                                jacobian_stats['sfc_max']
                            )
                            
                            # ========== Jacobian loss for clear-sky ==========
                            if n_clear > 0:
                                atm_jac_loss += criterion_l1(jac_atm_model_norm_tb, jac_atm_gt_norm_tb)
                                sfc_jac_loss += criterion_l1(jac_sfc_model_norm_tb, jac_sfc_gt_norm_tb)
                        
                        # Cloudy-sky samples
                        if cloudy_mask.any() and cloudy_mask.sum() > 0:
                            atm_cloudy = atm_input[cloudy_mask].requires_grad_(True)
                            sfc_cloudy = sfc_input[cloudy_mask].requires_grad_(True)
                            hydro_cloudy = hydro_input[cloudy_mask].requires_grad_(True)
                            
                            # Model Jacobians
                            jac_atm_model_norm_c, jac_sfc_model_norm_c, jac_hydro_model_norm = compute_model_jacobian(
                                model, atm_cloudy, sfc_cloudy, hydro_cloudy,
                                batch_size=batch_size, device=device
                            )

                            # Inverse scaling to physical space
                            jac_atm_model_raw_c = reverse_jacobian_batch(
                                jac_atm_model_norm_c,
                                scaler_tensors['atm'],
                                scaler_tensors['y'],
                            )
                            jac_sfc_model_raw_c = reverse_jacobian_sfc(
                                jac_sfc_model_norm_c,
                                scaler_tensors['sfc'],
                                scaler_tensors['y']
                            )
                            jac_hydro_model_raw = reverse_jacobian_batch(
                                jac_hydro_model_norm,
                                scaler_tensors['hydro'],
                                scaler_tensors['y'] 
                            )

                            # Ground truth for cloudy-sky
                            jac_atm_gt_cloudy = jac_atm_gt[cloudy_mask]
                            jac_sfc_gt_cloudy = jac_sfc_gt[cloudy_mask]
                            jac_cloud_gt_cloudy = jac_cloud_gt[cloudy_mask]

                            # Normalization using dataset statistics
                            jac_atm_model_norm_tb_c = safe_normalize_jacobian(
                                jac_atm_model_raw_c[:, :, :, :2],
                                jacobian_stats['atm_min'],
                                jacobian_stats['atm_max']
                            )
                            jac_sfc_model_norm_tb_c = safe_normalize_jacobian(
                                jac_sfc_model_raw_c[:, :, :3],
                                jacobian_stats['sfc_min'],
                                jacobian_stats['sfc_max']
                            )
                            jac_hydro_model_norm_tb = safe_normalize_jacobian(
                                jac_hydro_model_raw,
                                jacobian_stats['cloud_min'],
                                jacobian_stats['cloud_max']
                            )
                            
                            jac_atm_gt_norm_tb_c = safe_normalize_jacobian(
                                jac_atm_gt_cloudy,
                                jacobian_stats['atm_min'],
                                jacobian_stats['atm_max']
                            )
                            jac_sfc_gt_norm_tb_c = safe_normalize_jacobian(
                                jac_sfc_gt_cloudy,
                                jacobian_stats['sfc_min'],
                                jacobian_stats['sfc_max']
                            )
                            jac_cloud_gt_norm_tb = safe_normalize_jacobian(
                                jac_cloud_gt_cloudy,
                                jacobian_stats['cloud_min'],
                                jacobian_stats['cloud_max']
                            )
                            
                            # Jacobian loss for cloudy-sky
                            if n_cloudy > 0:
                                atm_jac_loss += criterion_l1(jac_atm_model_norm_tb_c, jac_atm_gt_norm_tb_c)
                                sfc_jac_loss += criterion_l1(jac_sfc_model_norm_tb_c, jac_sfc_gt_norm_tb_c)
                                hydro_jac_loss += criterion_l1(jac_hydro_model_norm_tb, jac_cloud_gt_norm_tb)

                # ========== Compute final Jacobian loss ==========
                # Atm and Sfc losses are averaged over total number of samples (clear+cloudy)
                final_atm_loss = atm_jac_loss / n_total if n_total > 0 else torch.tensor(0.0, device=device)
                final_sfc_loss = sfc_jac_loss / n_total if n_total > 0 else torch.tensor(0.0, device=device)
                # Hydro loss only exists for cloudy samples, average over cloudy count
                final_hydro_loss = hydro_jac_loss / n_cloudy if n_cloudy > 0 else torch.tensor(0.0, device=device)
                
                jac_loss = (final_atm_loss + final_sfc_loss + final_hydro_loss) * n_total

                total_loss = task_loss + lambda_jac * jac_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Record losses
                epoch_train_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_jac_loss += jac_loss.item() if isinstance(jac_loss, torch.Tensor) else jac_loss
                train_steps += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4e}",
                    'task': f"{task_loss.item():.4e}",
                    'jac': f"{jac_loss.item():.4e}" if isinstance(jac_loss, torch.Tensor) else f"{jac_loss:.4e}"
                })
        
        # Average training losses over epoch
        avg_train_loss = epoch_train_loss / train_steps
        avg_task_loss = epoch_task_loss / train_steps
        avg_jac_loss = epoch_jac_loss / train_steps
        
        train_losses.append(avg_train_loss)
        task_losses.append(avg_task_loss)
        jac_losses.append(avg_jac_loss)
        
        # ---------- Validation Phase (with Jacobian evaluation) ----------
        model.eval()
        val_loss = 0.0
        val_jac_loss = 0.0
        val_steps = 0
        
        with torch.enable_grad():  # we need gradients for Jacobian computation
            with tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="batch") as pbar:
                for batch in pbar:
                    # Move data
                    atm_input = batch['atm_input'].to(device)
                    sfc_input = batch['sfc_input'].to(device)
                    hydro_input = batch['hydro_input'].to(device)
                    output_gt = batch['output'].to(device)
                    jac_atm_gt = batch['jac_atm'].to(device)
                    jac_sfc_gt = batch['jac_sfc'].to(device)
                    jac_cloud_gt = batch['jac_cloud'].to(device)
                    is_cloudy = batch['is_cloudy'].to(device)
                    
                    clear_mask = (is_cloudy == 0)
                    cloudy_mask = (is_cloudy == 1)
                    n_clear_v = clear_mask.sum().item()
                    n_cloudy_v = cloudy_mask.sum().item()
                    n_total_v = atm_input.size(0)

                    radiance_pred = torch.zeros_like(output_gt)
                    
                    if clear_mask.any():
                        radiance_pred[clear_mask] = model(
                            atm_input[clear_mask], 
                            sfc_input[clear_mask]
                        )
                    
                    if cloudy_mask.any():
                        radiance_pred[cloudy_mask] = model(
                            atm_input[cloudy_mask], 
                            sfc_input[cloudy_mask], 
                            hydro_input[cloudy_mask]
                        )
                    
                    # Task loss
                    task_loss_val = criterion_mse(radiance_pred, output_gt)
                    val_loss += task_loss_val.item()
                    
                    # Jacobian loss for validation (similar to training)
                    atm_jac_loss_v = torch.tensor(0.0, device=device)
                    sfc_jac_loss_v = torch.tensor(0.0, device=device)
                    hydro_jac_loss_v = torch.tensor(0.0, device=device)
                    
                    # Clear-sky Jacobian
                    if clear_mask.any() and clear_mask.sum() > 0:
                        atm_clear = atm_input[clear_mask].requires_grad_(True)
                        sfc_clear = sfc_input[clear_mask].requires_grad_(True)
                        
                        jac_atm_model_norm, jac_sfc_model_norm, _ = compute_model_jacobian(
                            model, atm_clear, sfc_clear, None,
                            batch_size=batch_size*2, device=device
                        )
                        
                        jac_atm_model_raw = reverse_jacobian_batch(
                            jac_atm_model_norm,
                            scaler_tensors['atm'],
                            scaler_tensors['y']
                        )
                        jac_sfc_model_raw = reverse_jacobian_sfc(
                            jac_sfc_model_norm,
                            scaler_tensors['sfc'],
                            scaler_tensors['y']
                        )

                        jac_atm_gt_clear = jac_atm_gt[clear_mask]
                        jac_sfc_gt_clear = jac_sfc_gt[clear_mask]

                        jac_atm_model_norm_tb = safe_normalize_jacobian(
                            jac_atm_model_raw[:, :, :, :2],
                            jacobian_stats['atm_min'],
                            jacobian_stats['atm_max']
                        )
                        jac_sfc_model_norm_tb = safe_normalize_jacobian(
                            jac_sfc_model_raw[:, :, :3],
                            jacobian_stats['sfc_min'],
                            jacobian_stats['sfc_max']
                        )
                        jac_atm_gt_norm_tb = safe_normalize_jacobian(
                            jac_atm_gt_clear,
                            jacobian_stats['atm_min'],
                            jacobian_stats['atm_max']
                        )
                        jac_sfc_gt_norm_tb = safe_normalize_jacobian(
                            jac_sfc_gt_clear,
                            jacobian_stats['sfc_min'],
                            jacobian_stats['sfc_max']
                        )

                        atm_jac_loss_v += criterion_l1(jac_atm_model_norm_tb, jac_atm_gt_norm_tb)
                        sfc_jac_loss_v += criterion_l1(jac_sfc_model_norm_tb, jac_sfc_gt_norm_tb)

                    # Cloudy-sky Jacobian
                    if cloudy_mask.any() and cloudy_mask.sum() > 0:
                        atm_cloudy = atm_input[cloudy_mask].requires_grad_(True)
                        sfc_cloudy = sfc_input[cloudy_mask].requires_grad_(True)
                        hydro_cloudy = hydro_input[cloudy_mask].requires_grad_(True)
                        
                        jac_atm_model_norm_c, jac_sfc_model_norm_c, jac_hydro_model_norm = compute_model_jacobian(
                            model, atm_cloudy, sfc_cloudy, hydro_cloudy,
                            batch_size=batch_size*2, device=device
                        )
                        
                        jac_atm_model_raw_c = reverse_jacobian_batch(
                            jac_atm_model_norm_c,
                            scaler_tensors['atm'],
                            scaler_tensors['y']
                        )
                        jac_sfc_model_raw_c = reverse_jacobian_sfc(
                            jac_sfc_model_norm_c,
                            scaler_tensors['sfc'],
                            scaler_tensors['y']
                        )
                        jac_hydro_model_raw = reverse_jacobian_batch(
                            jac_hydro_model_norm,
                            scaler_tensors['hydro'],
                            scaler_tensors['y']
                        )

                        jac_atm_gt_cloudy = jac_atm_gt[cloudy_mask]
                        jac_sfc_gt_cloudy = jac_sfc_gt[cloudy_mask]
                        jac_cloud_gt_cloudy = jac_cloud_gt[cloudy_mask]

                        jac_atm_model_norm_tb_c = safe_normalize_jacobian(
                            jac_atm_model_raw_c[:, :, :, :2],
                            jacobian_stats['atm_min'],
                            jacobian_stats['atm_max']
                        )
                        jac_sfc_model_norm_tb_c = safe_normalize_jacobian(
                            jac_sfc_model_raw_c[:, :, :3],
                            jacobian_stats['sfc_min'],
                            jacobian_stats['sfc_max']
                        )
                        jac_hydro_model_norm_tb = safe_normalize_jacobian(
                            jac_hydro_model_raw,
                            jacobian_stats['cloud_min'],
                            jacobian_stats['cloud_max']
                        )

                        jac_atm_gt_norm_tb_c = safe_normalize_jacobian(
                            jac_atm_gt_cloudy,
                            jacobian_stats['atm_min'],
                            jacobian_stats['atm_max']
                        )
                        jac_sfc_gt_norm_tb_c = safe_normalize_jacobian(
                            jac_sfc_gt_cloudy,
                            jacobian_stats['sfc_min'],
                            jacobian_stats['sfc_max'] 
                        )
                        jac_cloud_gt_norm_tb = safe_normalize_jacobian(
                            jac_cloud_gt_cloudy,
                            jacobian_stats['cloud_min'],
                            jacobian_stats['cloud_max']
                        )

                        atm_jac_loss_v += criterion_l1(jac_atm_model_norm_tb_c, jac_atm_gt_norm_tb_c)
                        sfc_jac_loss_v += criterion_l1(jac_sfc_model_norm_tb_c, jac_sfc_gt_norm_tb_c)
                        hydro_jac_loss_v += criterion_l1(jac_hydro_model_norm_tb, jac_cloud_gt_norm_tb)
                        
                    # Average losses appropriately
                    final_atm_loss_v = atm_jac_loss_v / n_total_v if n_total_v > 0 else 0.0
                    final_sfc_loss_v = sfc_jac_loss_v / n_total_v if n_total_v > 0 else 0.0
                    final_hydro_loss_v = hydro_jac_loss_v / n_cloudy_v if n_cloudy_v > 0 else 0.0
                    
                    jac_loss_val = (final_atm_loss_v + final_sfc_loss_v + final_hydro_loss_v) * n_total_v
                    val_jac_loss += jac_loss_val.item()
                    val_steps += 1
                    
                    pbar.set_postfix({
                        'val_task': f"{task_loss_val.item():.4e}",
                        'val_jac': f"{jac_loss_val.item():.4e}"
                    })
        
        avg_val_loss = val_loss / val_steps
        avg_val_jac_loss = val_jac_loss / val_steps
        avg_val_total_loss = avg_val_loss + lambda_jac * avg_val_jac_loss
        val_losses.append(avg_val_total_loss)
        scheduler.step(avg_val_total_loss)
        
        print(f"Val Task Loss: {avg_val_loss:.6e} | Val Jac Loss: {avg_val_jac_loss:.6e}")
        
        # Save best model based on total validation loss (task + lambda_jac * jac)
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'task_loss': avg_task_loss,
                'jac_loss': avg_jac_loss,
                'scaler_info': scaler_info,
            }, save_path)
            patience_counter = 0
            print(f"✅ Best model saved. Validation loss: {best_val_loss:.6e}")
        else:
            patience_counter += 1
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.6e} (Task: {avg_task_loss:.6e}, Jac: {avg_jac_loss:.6e}) | "
              f"Val Loss: {avg_val_loss:.6e} | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"✅ Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n✅ Training finished! Best validation loss: {best_val_loss:.6e}")
    print(f"Model saved to: {save_path}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'task_losses': task_losses,
        'jac_losses': jac_losses,
        'best_val_loss': best_val_loss,
        'scaler_info': scaler_info
    }

    
if __name__ == "__main__":
    
    # Create model instance
    model = UnifiedAllSkyRTM_Transformer(hidden_dim=160, nhead=4, num_transformer_layers=3)
    
    # Data paths
    h5_path = './scalers/radiation_data_jacobian.h5'
    save_path = './scalers/unified_model_jacobian_finetuned.pth' 
    pretrained_path = './scalers/best_unified_model_ATT.pth'
    
    # Start training
    results = train_with_jacobian_constraint(
        model=model,
        h5_path=h5_path,
        save_path=save_path,
        pretrained_model_path=pretrained_path,
        lambda_jac=1e-5,   # Jacobian constraint weight
        jac_freq=1,        # Compute Jacobian loss every batch
        num_epochs=100,    # Number of epochs
        batch_size=256,    # Batch size
        learning_rate=1e-4,
        device='cuda'
    )