import numpy as np
from netCDF4 import Dataset as NetCDFDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset
from tqdm import tqdm
from time import time
import joblib
import os
import glob
import h5py
from GB_model import UnifiedAllSkyRTM_Transformer


class UnifiedRadiationDataset(Dataset):
    """
    Custom PyTorch Dataset for Unified All-Sky Radiative Transfer Model.
    Loads pre-processed data (atmospheric profiles, surface conditions, hydrometeors, and outputs) 
    from an HDF5 file into memory for fast training.
    """
    def __init__(self, h5_path, mode='train', max_samples=None):
        """
        Args:
            h5_path (str): Path to the HDF5 file containing the dataset.
            mode (str): 'train' or 'valid' to specify the data split.
            max_samples (int, optional): Maximum number of samples to load (for debugging/limited memory).
        """
        self.h5_path = h5_path
        self.mode = mode
        print(f"Loading {mode} data from {h5_path}...")

        # Load all required datasets from HDF5 into RAM (float32)
        with h5py.File(h5_path, 'r') as f:
            if mode == 'train':
                self.atm_clear = self._load_slice(f['atm_clear_train'], max_samples // 2)
                self.sfc_clear = self._load_slice(f['sfc_clear_train'], max_samples // 2)
                self.out_clear = self._load_slice(f['out_clear_train'], max_samples // 2)
                self.atm_cloudy = self._load_slice(f['atm_cloudy_train'], max_samples // 2)
                self.sfc_cloudy = self._load_slice(f['sfc_cloudy_train'], max_samples // 2)
                self.hydro_cloudy = self._load_slice(f['hydro_cloudy_train'], max_samples // 2)
                self.out_cloudy = self._load_slice(f['out_cloudy_train'], max_samples // 2)
            else:
                self.atm_clear = self._load_slice(f['atm_clear_valid'], max_samples // 2)
                self.sfc_clear = self._load_slice(f['sfc_clear_valid'], max_samples // 2)
                self.out_clear = self._load_slice(f['out_clear_valid'], max_samples // 2)
                self.atm_cloudy = self._load_slice(f['atm_cloudy_valid'], max_samples // 2)
                self.sfc_cloudy = self._load_slice(f['sfc_cloudy_valid'], max_samples // 2)
                self.hydro_cloudy = self._load_slice(f['hydro_cloudy_valid'], max_samples // 2)
                self.out_cloudy = self._load_slice(f['out_cloudy_valid'], max_samples // 2)

        self.clear_len = len(self.atm_clear)
        self.cloudy_len = len(self.atm_cloudy)
        self.total_len = self.clear_len + self.cloudy_len
        print(f"Loaded {self.clear_len} clear + {self.cloudy_len} cloudy = {self.total_len} total samples.")

    def _load_slice(self, dataset, max_count):
        """Safely slice the dataset and ensure data type is float32."""
        n = len(dataset)
        count = min(max_count, n) if max_count else n
        return dataset[:count].astype(np.float32)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """
        Returns a tuple based on the index.
        The dataset is structured as [Clear_Samples][Cloudy_Samples].
        If idx < clear_len -> return clear sample (with zero-padded hydro input).
        Else -> return cloudy sample.
        """
        if idx < self.clear_len:
            i = idx
            return (
                self.atm_clear[i],           # (37, 4)  Atmospheric profile
                self.sfc_clear[i],          # (3,)     Surface conditions
                np.zeros((36, 4), dtype=np.float32), # (36, 4) Hydrometeor placeholder (zero-filled for clear)
                self.out_clear[i],          # (22,)    Target output (e.g., Brightness Temperatures)
                0                           # Label: 0 for clear
            )
        else:
            i = idx - self.clear_len
            return (
                self.atm_cloudy[i],         # (37, 4)
                self.sfc_cloudy[i],         # (3,)
                self.hydro_cloudy[i],       # (36, 4)  Actual hydrometeor data (CLWC, CIWC, etc.)
                self.out_cloudy[i],         # (22,)
                1                           # Label: 1 for cloudy
            )


# =============================
# 3. Training Main Function
# =============================
def train_unified_model(model, h5_path, save_path):
    """
    Main training loop for the unified radiation model.
    Handles data loading, mixed precision training, validation, and early stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # -------------------------------
    # Hyperparameters
    # -------------------------------
    max_train_samples = 2400000  # Limit for training data
    max_val_samples = 980000     # Limit for validation data
    train_batch_size = 256
    val_batch_size = 1024
    num_workers = 8              # Number of subprocesses for data loading
    prefetch_factor = 4          # Number of samples loaded in advance by each worker
    num_epochs = 1000
    patience = 25                # For early stopping

    # -------------------------------
    # Dataset Construction
    # -------------------------------
    print("Loading datasets...")
    train_dataset = UnifiedRadiationDataset(
        h5_path, 
        mode='train', 
        max_samples=max_train_samples 
    )
    val_dataset = UnifiedRadiationDataset(
        h5_path, 
        mode='valid', 
        max_samples=max_val_samples 
    )

    # DataLoader with optimizations: pin_memory, persistent workers, prefetch
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,           # Faster GPU transfer
        persistent_workers=True,   # Keep workers alive between epochs
        prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    print(f"Train loader: {len(train_loader)} batches per epoch")
    print(f"Val loader: {len(val_loader)} batches per epoch")

    # -------------------------------
    # Model, Loss, Optimizer
    # -------------------------------
    model = model.to(device)
    
    # Optional: Load pre-trained weights for fine-tuning
    # model.load_state_dict(torch.load('./best_unified_model.pth')) 
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Reduce learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Gradient scaler for mixed precision training (FP16)
    scaler = torch.amp.GradScaler("cuda") 

    # -------------------------------
    # Training Loop
    # -------------------------------
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print("\n Starting training...")
    
    for epoch in range(num_epochs):
        start_time = time()

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        # Progress bar using tqdm
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch") as pbar:
            for X_atm, X_sfc, X_hydro, y_true, is_cloudy in pbar:
                # Transfer data to GPU
                X_atm = X_atm.to(device, non_blocking=True)
                X_sfc = X_sfc.to(device, non_blocking=True)
                X_hydro = X_hydro.to(device, non_blocking=True)
                y_true = y_true.to(device, non_blocking=True)
                is_cloudy = is_cloudy.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Mixed Precision Context
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = torch.zeros(y_true.size(0), 22, device=device, dtype=torch.float16)
                    
                    # Create masks to separate clear and cloudy samples in the batch
                    clear_mask = (is_cloudy == 0)
                    cloudy_mask = (is_cloudy == 1)

                    # Forward pass for clear-sky samples (without hydro input)
                    if clear_mask.any():
                        out_clear = model(X_atm[clear_mask], X_sfc[clear_mask])
                        outputs[clear_mask] = out_clear

                    # Forward pass for cloudy-sky samples (with hydro input)
                    if cloudy_mask.any():
                        out_cloudy = model(X_atm[cloudy_mask], X_sfc[cloudy_mask], X_hydro[cloudy_mask])
                        outputs[cloudy_mask] = out_cloudy

                    loss = criterion(outputs, y_true)

                # Backward pass using scaled gradients to prevent underflow
                scaler.scale(loss).backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_steps += 6
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="batch") as pbar:
                for X_atm, X_sfc, X_hydro, y_true, is_cloudy in pbar:
                    X_atm = X_atm.to(device, non_blocking=True)
                    X_sfc = X_sfc.to(device, non_blocking=True)
                    X_hydro = X_hydro.to(device, non_blocking=True)
                    y_true = y_true.to(device, non_blocking=True)
                    is_cloudy = is_cloudy.to(device, non_blocking=True)

                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        outputs = torch.zeros(y_true.size(0), 22, device=device, dtype=torch.float16)
                        clear_mask = (is_cloudy == 0)
                        cloudy_mask = (is_cloudy == 1)

                        if clear_mask.any():
                            outputs[clear_mask] = model(X_atm[clear_mask], X_sfc[clear_mask])
                        if cloudy_mask.any():
                            outputs[cloudy_mask] = model(X_atm[cloudy_mask], X_sfc[cloudy_mask], X_hydro[cloudy_mask])

                        loss = criterion(outputs, y_true)
                    val_loss += loss.item()
                    val_steps += 1
                    pbar.set_postfix(val_loss=f"{loss.item():.6f}")

        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)

        # --- Learning Rate Scheduler & Early Stopping ---
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"✅ Best model saved. Val Loss: {best_val_loss:.6e}")
        else:
            epochs_no_improve += 1

        epoch_time = time() - start_time
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6e} | "
              f"Val Loss: {avg_val_loss:.6e} | Time: {epoch_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if epochs_no_improve >= patience:
            print(f"✅ Early stopping at epoch {epoch+1}")
            break

    print(f"Training completed. Best validation loss: {best_val_loss:.6e}")
    return model, train_losses, val_losses


if __name__ == "__main__":
    print("=== Unified All-Sky Radiative Transfer Model Training ===")
    
    # Set number of threads for PyTorch (optional for CPU ops)
    torch.set_num_threads(8) 
    
    # --- Model Configuration ---
    # Note: The original code imported UnifiedAllSkyRTM_Transformer
    # You may need to adjust this line based on your actual model class
    model = UnifiedAllSkyRTM_Transformer(hidden_dim=160, nhead=4, num_transformer_layers=3)
    
    # Data and Save Paths
    h5_path = './scalers/radiation_data.h5'
    
    # Please ensure the filename matches your actual architecture.
    save_path = './scalers/best_unified_model_Transformer.pth' 
    
    # Start Training
    model, train_losses, val_losses = train_unified_model(model, h5_path, save_path)
    print("=== ✅ Training Complete ===")