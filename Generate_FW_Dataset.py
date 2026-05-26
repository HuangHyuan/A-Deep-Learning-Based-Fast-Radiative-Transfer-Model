import numpy as np
from netCDF4 import Dataset as NetCDFDataset
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import glob
import h5py

# -------------------------------
# 1. Data Loading Function
# -------------------------------
def load_single_data(sim_path, nc_path):
    """
    Load and process a single pair of simulation (txt) and IFS model (nc) data files.
    
    Args:
        sim_path (str): Path to the simulation output .txt file (BT data).
        nc_path (str): Path to the IFS atmospheric input .nc file.
        
    Returns:
        tuple: Scaled input arrays (atm, sfc, hydro) and output targets.
    """
    
    # --- Read Simulation Output (Target) ---
    # Typically contains simulated brightness temperatures
    output_sim = np.loadtxt(sim_path)
    
    # --- Read IFS NetCDF Input (Features) ---
    nc_file = NetCDFDataset(nc_path, 'r')
    
    # Atmospheric profile variables (37 vertical levels)
    Level_H2O = nc_file.variables['Level_H2O'][:]      # Water vapor
    Level_O3 = nc_file.variables['Level_O3'][:]        # Ozone
    Level_Pressure = nc_file.variables['Level_Pressure'][:] # Pressure
    Level_Temperature = nc_file.variables['Level_Temperature'][:] # Temperature
    
    # Surface & Integrated variables
    Temperature_2M = nc_file.variables['Temperature_2M'][:] # 2m Temperature
    H2O_2M = nc_file.variables['H2O_2M'][:]               # 2m Water vapor
    Surface_Pressure = nc_file.variables['Surface_Pressure'][:] # Surface pressure
    TCLW = nc_file.variables['TCLW'][:]                   # Total Column Liquid Water
    TCRW = nc_file.variables['TCRW'][:]                   # Total Column Rain Water
    
    # Cloud hydrometeors (37 levels)
    Level_CLWC = nc_file.variables['Level_CLWC'][:]      # Cloud Liquid Water Content
    Level_CIWC = nc_file.variables['Level_CIWC'][:]      # Cloud Ice Water Content
    Level_CRWC = nc_file.variables['Level_CRWC'][:]      # Cloud Rain Water Content
    Level_CSWC = nc_file.variables['Level_CSWC'][:]      # Cloud Snow Water Content
    
    # Geopotential height for layer thickness calculation
    geopotential_height = nc_file.variables['geopotential_height'][:] 
    
    nc_file.close()
    
    # Transpose profile variables if they are in (37, N) format
    if Level_H2O.shape[0] == 37:
        # Store original shape info for debugging
        original_shape = Level_H2O.shape
        
        # Transpose all profile variables
        Level_H2O = Level_H2O.T
        Level_O3 = Level_O3.T
        Level_Pressure = Level_Pressure.T
        Level_Temperature = Level_Temperature.T
        geopotential_height = geopotential_height.T
        
        # Transpose hydrometeors
        Level_CIWC = Level_CIWC.T
        Level_CLWC = Level_CLWC.T
        Level_CRWC = Level_CRWC.T
        Level_CSWC = Level_CSWC.T

    # --- Calculate Layer Thickness (dz) and Mid-layer Values ---
    dz = geopotential_height[:, :-1] - geopotential_height[:, 1:] # Shape: (N, 36)
    
    # Calculate mid-layer values for hydrometeors
    # We average the value between the top and bottom of the layer
    CLWC = (Level_CLWC[:, :-1] + Level_CLWC[:, 1:]) / 2 # (N, 36)
    CIWC = (Level_CIWC[:, :-1] + Level_CIWC[:, 1:]) / 2 # (N, 36)
    CRWC = (Level_CRWC[:, :-1] + Level_CRWC[:, 1:]) / 2 # (N, 36)
    CSWC = (Level_CSWC[:, :-1] + Level_CSWC[:, 1:]) / 2 # (N, 36)
    
    # --- Input Feature Construction ---
    # 1. Atmospheric Profiles (atm_input): Contains varying profiles
    # Shape: (N_samples, 37, 3) -> [H2O, Temperature, Pressure]
    atm_input = np.concatenate([
        Level_H2O[:, :, np.newaxis],
        Level_Temperature[:, :, np.newaxis],
        Level_Pressure[:, :, np.newaxis]
    ], axis=-1)

    # 2. Surface Conditions (sfc_input): Contains 2m variables + integrated columns
    # Shape: (N_samples, 5) -> [T2m, H2O_2m, Psurf, TCLW, TCRW]
    sfc_input = np.concatenate([
        Temperature_2M[:, np.newaxis],
        H2O_2M[:, np.newaxis],
        Surface_Pressure[:, np.newaxis],
        TCLW[:, np.newaxis],
        TCRW[:, np.newaxis],
    ], axis=-1)

    # --- Hydrometeor Processing (Cloudy Data Only) ---
    # Zero out values below a threshold to treat them as strictly zero
    # This helps the model learn the "clear sky" boundary
    threshold = 1e-8
    CLWC[CLWC < threshold] = 0
    CRWC[CRWC < threshold] = 0
    CSWC[CSWC < threshold] = 0
    CIWC[CIWC < threshold] = 0

    # Construct hydro_input: Multiply by dz to get column density per layer
    # Shape: (N_samples, 36, 4) -> [CLWC*dz, CRWC*dz, CSWC*dz, CIWC*dz]
    hydro_input = np.concatenate([
        (CLWC * dz)[:, :, np.newaxis],
        (CRWC * dz)[:, :, np.newaxis],
        (CSWC * dz)[:, :, np.newaxis],
        (CIWC * dz)[:, :, np.newaxis],
    ], axis=-1)

    # Output: Simulated observations
    # Shape: (N_samples, 22) -> e.g., 22 satellite channels
    output = output_sim 

    return atm_input, sfc_input, hydro_input, output

# -------------------------------
# 2. Batch Loading Functions
# -------------------------------
def load_all_data(sim_dir, nc_dir, prefix_sim, prefix_nc):
    """
    Load all data files from specified directories matching the given prefixes.
    """
    sim_files = sorted(glob.glob(os.path.join(sim_dir, f"{prefix_sim}*.txt")))
    nc_files = sorted(glob.glob(os.path.join(nc_dir, f"{prefix_nc}*.nc")))
    
    assert len(sim_files) == len(nc_files), "File count mismatch between simulation and NetCDF directories!"
    
    all_atm, all_sfc, all_hydro, all_out = [], [], [], []
    
    for sim_path, nc_path in zip(sim_files, nc_files):
        print(f"Loading: {sim_path} & {nc_path}")
        atm, sfc, hydro, out = load_single_data(sim_path, nc_path)
        all_atm.append(atm)
        all_sfc.append(sfc)
        all_hydro.append(hydro)
        all_out.append(out)
        
    return (
        np.concatenate(all_atm, axis=0),
        np.concatenate(all_sfc, axis=0),
        np.concatenate(all_hydro, axis=0),
        np.concatenate(all_out, axis=0)
    )

# -------------------------------
# 3. Load Training and Validation Datasets
# -------------------------------
print("Loading datasets...")

# --- Training Data (2023) ---
# Clear Sky Training
atm_clear_train, sfc_clear_train, _, out_clear_train = load_all_data(
    sim_dir='/root/autodl-tmp/clear2023',
    nc_dir='/root/autodl-tmp/clear_sample',
    prefix_sim='clear_2023_',
    prefix_nc='ifs_clear_2023-'
)

# Cloudy Sky Training (contains hydro_input)
atm_cloudy_train, sfc_cloudy_train, hydro_cloudy_train, out_cloudy_train = load_all_data(
    sim_dir='/root/autodl-tmp/cloudy2023',
    nc_dir='/root/autodl-tmp/cloudy_sample',
    prefix_sim='cloudy_2023_',
    prefix_nc='ifs_cloudy_2023-'
)

# --- Validation Data (2024) ---
# Clear Sky Validation
atm_clear_valid, sfc_clear_valid, _, out_clear_valid = load_all_data(
    sim_dir='/root/autodl-tmp/clear202401',
    nc_dir='/root/autodl-tmp/Valid_clear_sample',
    prefix_sim='clear_2024_',
    prefix_nc='ifs_clear_2024-'
)

# Cloudy Sky Validation
atm_cloudy_valid, sfc_cloudy_valid, hydro_cloudy_valid, out_cloudy_valid = load_all_data(
    sim_dir='/root/autodl-tmp/cloudy202401',
    nc_dir='/root/autodl-tmp/Valid_cloudy_sample',
    prefix_sim='cloudy_2024_',
    prefix_nc='ifs_cloudy_2024-'
)

# --- Data Cleaning: Set Cloud Water to Zero for Clear Sky ---
sfc_clear_train[:, -2:] = 0
sfc_clear_valid[:, -2:] = 0

print(f"Train Clear: atm={atm_clear_train.shape}, sfc={sfc_clear_train.shape}, out={out_clear_train.shape}")
print(f"Train Cloudy: atm={atm_cloudy_train.shape}, sfc={sfc_cloudy_train.shape}, hydro={hydro_cloudy_train.shape}, out={out_cloudy_train.shape}")
print(f"Valid Clear: atm={atm_clear_valid.shape}, sfc={sfc_clear_valid.shape}, out={out_clear_valid.shape}")
print(f"Valid Cloudy: atm={atm_cloudy_valid.shape}, sfc={sfc_cloudy_valid.shape}, hydro={hydro_cloudy_valid.shape}, out={out_cloudy_valid.shape}")

# -------------------------------
# 4. Data Standardization (Fitting Scalers)
# -------------------------------
print("Fitting Scalers...")

# --- 1. Atmospheric Input Scaler (Layer-wise) ---
scaler_atm_global = []
atm_clear_train_scaled = np.zeros_like(atm_clear_train)
atm_cloudy_train_scaled = np.zeros_like(atm_cloudy_train)
atm_clear_valid_scaled = np.zeros_like(atm_clear_valid)
atm_cloudy_valid_scaled = np.zeros_like(atm_cloudy_valid)

for k in range(atm_clear_train.shape[1]): # Loop over 37 levels
    # Combine clear and cloudy data for fitting
    layer_data_train = np.concatenate([
        atm_clear_train[:, k, :],
        atm_cloudy_train[:, k, :]
    ], axis=0)
    
    scaler_k = MinMaxScaler().fit(layer_data_train)
    scaler_atm_global.append(scaler_k)
    
    # Transform data for this specific level
    atm_clear_train_scaled[:, k, :] = scaler_k.transform(atm_clear_train[:, k, :])
    atm_cloudy_train_scaled[:, k, :] = scaler_k.transform(atm_cloudy_train[:, k, :])
    atm_clear_valid_scaled[:, k, :] = scaler_k.transform(atm_clear_valid[:, k, :])
    atm_cloudy_valid_scaled[:, k, :] = scaler_k.transform(atm_cloudy_valid[:, k, :])

# --- 2. Surface Input Scaler (Global) ---
# Standardize all surface features together
sfc_train_all = np.concatenate([sfc_clear_train, sfc_cloudy_train], axis=0)
scaler_sfc_global = MinMaxScaler().fit(sfc_train_all)

sfc_clear_train_scaled = scaler_sfc_global.transform(sfc_clear_train)
sfc_cloudy_train_scaled = scaler_sfc_global.transform(sfc_cloudy_train)
sfc_clear_valid_scaled = scaler_sfc_global.transform(sfc_clear_valid)
sfc_cloudy_valid_scaled = scaler_sfc_global.transform(sfc_cloudy_valid)

# --- 3. Hydrometeor Input Scaler (Cloudy Only, Layer-wise) ---
# Similar to atmosphere, standardize each layer independently
scaler_hydro_cloudy = []
hydro_cloudy_train_scaled = np.zeros_like(hydro_cloudy_train)
hydro_cloudy_valid_scaled = np.zeros_like(hydro_cloudy_valid)

for k in range(hydro_cloudy_train.shape[1]): # Loop over 36 layers
    scaler_k = MinMaxScaler().fit(hydro_cloudy_train[:, k, :])
    scaler_hydro_cloudy.append(scaler_k)
    
    hydro_cloudy_train_scaled[:, k, :] = scaler_k.transform(hydro_cloudy_train[:, k, :])
    hydro_cloudy_valid_scaled[:, k, :] = scaler_k.transform(hydro_cloudy_valid[:, k, :])

# --- 4. Output Scaler (Target Variable) ---
# Standardize the simulation output (e.g., Brightness Temperatures)
y_train_all = np.concatenate([out_clear_train, out_cloudy_train], axis=0)
scaler_y = MinMaxScaler().fit(y_train_all)

out_clear_train_scaled = scaler_y.transform(out_clear_train)
out_cloudy_train_scaled = scaler_y.transform(out_cloudy_train)
out_clear_valid_scaled = scaler_y.transform(out_clear_valid)
out_cloudy_valid_scaled = scaler_y.transform(out_cloudy_valid)

# -------------------------------
# 5. Convert to PyTorch Tensors
# -------------------------------
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

# Training sets
atm_clear_train_t = to_tensor(atm_clear_train_scaled)
sfc_clear_train_t = to_tensor(sfc_clear_train_scaled)
out_clear_train_t = to_tensor(out_clear_train_scaled)

atm_cloudy_train_t = to_tensor(atm_cloudy_train_scaled)
sfc_cloudy_train_t = to_tensor(sfc_cloudy_train_scaled)
hydro_cloudy_train_t = to_tensor(hydro_cloudy_train_scaled)
out_cloudy_train_t = to_tensor(out_cloudy_train_scaled)

# Validation sets
atm_clear_valid_t = to_tensor(atm_clear_valid_scaled)
sfc_clear_valid_t = to_tensor(sfc_clear_valid_scaled)
out_clear_valid_t = to_tensor(out_clear_valid_scaled)

atm_cloudy_valid_t = to_tensor(atm_cloudy_valid_scaled)
sfc_cloudy_valid_t = to_tensor(sfc_cloudy_valid_scaled)
hydro_cloudy_valid_t = to_tensor(hydro_cloudy_valid_scaled)
out_cloudy_valid_t = to_tensor(out_cloudy_valid_scaled)

# -------------------------------
# 6. Save Scalers and Processed Data
# -------------------------------
os.makedirs('./scalers', exist_ok=True)

# Save the scaler objects for inference
joblib.dump(scaler_atm_global, './scalers/scaler_atm_global.pkl')
joblib.dump(scaler_sfc_global, './scalers/scaler_sfc_global.pkl')
joblib.dump(scaler_hydro_cloudy, './scalers/scaler_hydro_cloudy.pkl')
joblib.dump(scaler_y, './scalers/scaler_y.pkl')

print("✅ Data loading and standardization completed. Scalers saved to ./scalers/")

# --- Extract Min/Max Info for Reference ---
def extract_scaler_info(scaler_atm_list, scaler_sfc, scaler_hydro_list, scaler_y):
    """Extract min/max values from scalers for documentation or C++ integration."""
    # Atmospheric
    atm_max_list = [scaler.data_max_ for scaler in scaler_atm_list]
    atm_min_list = [scaler.data_min_ for scaler in scaler_atm_list]
    
    # Surface
    sfc_max = scaler_sfc.data_max_
    sfc_min = scaler_sfc.data_min_
    
    # Hydrometeors
    hydro_max_list = [scaler.data_max_ for scaler in scaler_hydro_list]
    hydro_min_list = [scaler.data_min_ for scaler in scaler_hydro_list]
    
    # Output
    y_max = scaler_y.data_max_
    y_min = scaler_y.data_min_
    
    return {
        'atm': {
            'max_array': np.array(atm_max_list), # Shape: (37, 3)
            'min_array': np.array(atm_min_list), # Shape: (37, 3)
        },
        'sfc': {
            'max': sfc_max, # Shape: (5,)
            'min': sfc_min, # Shape: (5,)
        },
        'hydro': {
            'max_array': np.array(hydro_max_list), # Shape: (36, 4)
            'min_array': np.array(hydro_min_list), # Shape: (36, 4)
        },
        'y': {
            'max': y_max, # Shape: (22,)
            'min': y_min, # Shape: (22,)
        }
    }

# Extract and save scaler info
scaler_info = extract_scaler_info(scaler_atm_global, scaler_sfc_global, scaler_hydro_cloudy, scaler_y)
joblib.dump(scaler_info, './scalers/scaler_info.pkl')

# --- Save to HDF5 ---
print("Saving data to HDF5 file...")
with h5py.File('./scalers/radiation_data.h5', 'w') as h5_file:
    # Training Data
    h5_file.create_dataset('atm_clear_train', data=atm_clear_train_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('sfc_clear_train', data=sfc_clear_train_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('out_clear_train', data=out_clear_train_scaled, compression='gzip', compression_opts=6)
    
    h5_file.create_dataset('atm_cloudy_train', data=atm_cloudy_train_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('sfc_cloudy_train', data=sfc_cloudy_train_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('hydro_cloudy_train', data=hydro_cloudy_train_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('out_cloudy_train', data=out_cloudy_train_scaled, compression='gzip', compression_opts=6)
    
    # Validation Data
    h5_file.create_dataset('atm_clear_valid', data=atm_clear_valid_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('sfc_clear_valid', data=sfc_clear_valid_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('out_clear_valid', data=out_clear_valid_scaled, compression='gzip', compression_opts=6)
    
    h5_file.create_dataset('atm_cloudy_valid', data=atm_cloudy_valid_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('sfc_cloudy_valid', data=sfc_cloudy_valid_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('hydro_cloudy_valid', data=hydro_cloudy_valid_scaled, compression='gzip', compression_opts=6)
    h5_file.create_dataset('out_cloudy_valid', data=out_cloudy_valid_scaled, compression='gzip', compression_opts=6)

print("✅ Data preprocessing completed and saved as HDF5 file")