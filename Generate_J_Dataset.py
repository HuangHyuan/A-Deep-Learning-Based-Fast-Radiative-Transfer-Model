import numpy as np
import torch
import netCDF4 as nc
import h5py
import glob
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_single_data(sim_path, nc_path):
    """
    Load single data file pair (simulation output + IFS atmospheric profile).
    
    Args:
        sim_path (str): Path to the simulation output .txt file.
        nc_path (str): Path to the IFS atmospheric profile .nc file.
    
    Returns:
        tuple: (atm_input, sfc_input, hydro_input, output)
    """
    # Load simulation output (e.g., Brightness Temperatures)
    output_sim = np.loadtxt(sim_path)  # Shape: (N_profiles, 22_channels)

    # Read NetCDF atmospheric data
    nc_file = nc.Dataset(nc_path, 'r')
    
    # Atmospheric level variables (37 vertical levels)
    Level_H2O = nc_file.variables['Level_H2O'][:]          # Water vapor
    Level_O3 = nc_file.variables['Level_O3'][:]            # Ozone
    Level_Pressure = nc_file.variables['Level_Pressure'][:] # Pressure
    Level_Temperature = nc_file.variables['Level_Temperature'][:] # Temperature
    
    # Surface and column variables
    Temperature_2M = nc_file.variables['Temperature_2M'][:]  # 2m Temperature
    H2O_2M = nc_file.variables['H2O_2M'][:]                # 2m Water vapor
    Surface_Pressure = nc_file.variables['Surface_Pressure'][:] # Surface pressure
    
    # Cloud variables (37 levels)
    Level_CLWC = nc_file.variables['Level_CLWC'][:]        # Cloud liquid water content
    Level_CIWC = nc_file.variables['Level_CIWC'][:]        # Cloud ice water content
    Level_CRWC = nc_file.variables['Level_CRWC'][:]        # Cloud Rain water content
    Level_CSWC = nc_file.variables['Level_CSWC'][:]        # Cloud Snow water content
    
    # Geopotential height (used for layer thickness calculation)
    geopotential_height = nc_file.variables['geopotential_height'][:] 
    
    # Total Column Water (used for surface feature)
    TCLW = nc_file.variables['TCLW'][:]                    # Total column liquid water
    TCRW = nc_file.variables['TCRW'][:]                    # Total column rain water
    
    nc_file.close()

    # Transpose processing: Convert from (37, N) to (N, 37) for consistency
    variables = [Level_H2O, Level_O3, Level_Pressure, Level_Temperature, 
                geopotential_height, Level_CIWC, Level_CLWC, Level_CRWC, Level_CSWC]
    
    for i, var in enumerate(variables):
        if var.shape[0] == 37:  # Only transpose if first dim is 37 (levels)
            variables[i] = var.T
    
    (Level_H2O, Level_O3, Level_Pressure, Level_Temperature, geopotential_height,
     Level_CIWC, Level_CLWC, Level_CRWC, Level_CSWC) = variables

    # Calculate layer thickness (dz) and mid-layer values for cloud variables
    # geopotential_height: (N_profiles, 37) -> dz: (N_profiles, 36)
    dz = geopotential_height[:, :-1] - geopotential_height[:, 1:] 
    
    # Calculate mid-layer values for cloud hydrometeors
    CLWC = (Level_CLWC[:, :-1] + Level_CLWC[:, 1:]) / 2  # (N, 36)
    CIWC = (Level_CIWC[:, :-1] + Level_CIWC[:, 1:]) / 2
    CRWC = (Level_CRWC[:, :-1] + Level_CRWC[:, 1:]) / 2
    CSWC = (Level_CSWC[:, :-1] + Level_CSWC[:, 1:]) / 2

    # Construct Input Features
    # Atmospheric Input: (N_profiles, 37_levels, 3_features) [H2O, Temp, Pressure]
    atm_input = np.concatenate([
        Level_H2O[:, :, np.newaxis],
        Level_Temperature[:, :, np.newaxis],
        Level_Pressure[:, :, np.newaxis]
    ], axis=-1)

    # Surface Input: (N_profiles, 5_features)
    sfc_input = np.concatenate([
        Temperature_2M[:, np.newaxis],    # 2m Temp
        H2O_2M[:, np.newaxis],            # 2m H2O
        Surface_Pressure[:, np.newaxis],  # Surface Pressure
        TCLW[:, np.newaxis],              # Total Column Liquid Water 
        TCRW[:, np.newaxis]               # Total Column Rain Water 
    ], axis=-1)

    # Clean up near-zero values in cloud variables to enforce sparsity
    CLWC[CLWC < 1e-8] = 0
    CRWC[CRWC < 1e-8] = 0
    CSWC[CSWC < 1e-8] = 0
    CIWC[CIWC < 1e-8] = 0

    # Hydro Input (Cloud hydrometeors): (N_profiles, 36_layers, 4_features)
    # Using mass loading (content * thickness) as feature
    hydro_input = np.concatenate([
        (CLWC * dz)[:, :, np.newaxis], # Liquid
        (CRWC * dz)[:, :, np.newaxis], # Rain
        (CSWC * dz)[:, :, np.newaxis], # Snow
        (CIWC * dz)[:, :, np.newaxis]  # Ice
    ], axis=-1)

    output = output_sim  # Model output target (N_profiles, 22)

    return atm_input, sfc_input, hydro_input, output


def load_Jm_data(nc_path):
    """
    Load Jacobian Matrix (Sensitivity Matrix) data.
    
    Args:
        nc_path (str): Path to the Jacobian .nc file.
    
    Returns:
        tuple: (jac_atm, jac_sfc, jac_cloud)
    """
    ds = nc.Dataset(nc_path, 'r')
    
    # Atmospheric Jacobians: (N, 37_levels, 22_channels)
    jac_T = ds.variables['jacobian_temperature'][:].transpose((2, 0, 1)) 
    jac_Q = ds.variables['jacobian_h2o'][:].transpose((2, 0, 1)) 
    
    # Surface Jacobian: (N, 22_channels, 3_features) -> Transpose to (N, 3, 22)
    jac_surf = ds.variables['jacobian_surface'][:].transpose((2, 1, 0)) 
    
    # Cloud Jacobian (if exists)
    if 'jacobian_cloud' in ds.variables:
        # (N, 36_layers, 22_channels, 4_hydrometeors) 
        jac_cloud = ds.variables['jacobian_cloud'][:].transpose((3, 1, 2, 0)) 
    else:
        # Create zero matrix for clear-sky cases
        n_profiles = jac_T.shape[0]
        jac_cloud = np.zeros((n_profiles, 36, 22, 4), dtype=np.float32)
    
    ds.close()

    # Combine atmospheric Jacobians into a single tensor: (N, 37, 22, 2)
    # Last dimension: [dR/dQ, dR/dT]
    jac_atm = np.concatenate([
        jac_Q[:, :, :, np.newaxis], 
        jac_T[:, :, :, np.newaxis]
    ], axis=-1)
    
    return jac_atm, jac_surf, jac_cloud


# ============================================================================
# Batch Data Loading Functions
# ============================================================================

def load_day_data_bath(sim_dir, nc_dir, prefix_sim, prefix_nc, data_type='clear'):
    """
    Load atmospheric feature data in batch.
    
    Args:
        sim_dir (str): Directory for simulation output files.
        nc_dir (str): Directory for IFS NetCDF files.
        prefix_sim (str): Filename prefix for simulation files.
        prefix_nc (str): Filename prefix for NetCDF files.
        data_type (str): 'clear' or 'cloudy'.
    
    Returns:
        tuple: Batched numpy arrays for atm, sfc, hydro, out.
    """
    # Find files (specifically for the 16th in this script)
    sim_files = sorted(glob.glob(os.path.join(sim_dir, f"{prefix_sim}*16.txt")))
    nc_files = sorted(glob.glob(os.path.join(nc_dir, f"{prefix_nc}*16.nc")))
    
    print(f"Found {len(sim_files)} files")

    all_atm, all_sfc, all_hydro, all_out = [], [], [], []

    for sim_path, nc_path in zip(sim_files, nc_files):
        try:
            atm, sfc, hydro, out = load_single_data(sim_path, nc_path)
            all_atm.append(atm)
            all_sfc.append(sfc)
            all_hydro.append(hydro)
            all_out.append(out)
        except Exception as e:
            print(f"Error loading file {sim_path}: {e}")
            continue

    if not all_atm:
        raise ValueError(f"No {data_type} data found")

    # Concatenate all data
    atm_data = np.concatenate(all_atm, axis=0)
    sfc_data = np.concatenate(all_sfc, axis=0)
    hydro_data = np.concatenate(all_hydro, axis=0) if data_type == 'cloudy' else None
    out_data = np.concatenate(all_out, axis=0)

    print(f"Loading completed: {atm_data.shape[0]} profiles")
    return atm_data, sfc_data, hydro_data, out_data


def load_J_data_batch(jac_dir, data_type='clear'):
    """
    Load Jacobian data in batch.
    """
    jac_files = sorted(glob.glob(os.path.join(jac_dir, "*.nc")))
    print(f"Found {len(jac_files)} Jacobian files")

    all_jac_atm, all_jac_sfc, all_jac_cloud = [], [], []

    for jac_path in jac_files:
        try:
            jac_atm, jac_sfc, jac_cloud = load_Jm_data(jac_path)
            all_jac_atm.append(jac_atm)
            all_jac_sfc.append(jac_sfc)
            all_jac_cloud.append(jac_cloud)
        except Exception as e:
            print(f"Error loading Jacobian file {jac_path}: {e}")
            continue

    if not all_jac_atm:
        raise ValueError(f"No {data_type} Jacobian data found")

    jac_atm_data = np.concatenate(all_jac_atm, axis=0)
    jac_sfc_data = np.concatenate(all_jac_sfc, axis=0)
    jac_cloud_data = np.concatenate(all_jac_cloud, axis=0) if data_type == 'cloudy' else None

    print(f"Loading completed: {jac_atm_data.shape[0]} profiles")
    return jac_atm_data, jac_sfc_data, jac_cloud_data


# ============================================================================
# Main Program: Build Dataset
# ============================================================================

print("=" * 80)
print("Starting Dataset Construction (Atmospheric Features + Jacobian Matrix)")
print("=" * 80)

# ============================================================================
# 1. Load Atmospheric Features
# ============================================================================

print("\n1. Loading Atmospheric Feature Data...")

# --- Training Set ---
print("\nLoading Training Set...")
atm_clear_train, sfc_clear_train, _, out_clear_train = load_day_data_bath(
    sim_dir='/root/autodl-tmp/clear2023',
    nc_dir='/root/autodl-tmp/clear_sample',
    prefix_sim='clear_2023_',
    prefix_nc='ifs_clear_2023-',
    data_type='clear'
)

atm_cloudy_train, sfc_cloudy_train, hydro_cloudy_train, out_cloudy_train = load_day_data_bath(
    sim_dir='/root/autodl-tmp/cloudy2023',
    nc_dir='/root/autodl-tmp/cloudy_sample',
    prefix_sim='cloudy_2023_',
    prefix_nc='ifs_cloudy_2023-',
    data_type='cloudy'
)

# --- Validation Set ---
print("\nLoading Validation Set...")
atm_clear_valid, sfc_clear_valid, _, out_clear_valid = load_day_data_bath(
    sim_dir='/root/autodl-tmp/clear2024',
    nc_dir='/root/autodl-tmp/clear_sample',
    prefix_sim='clear_2024_',
    prefix_nc='ifs_clear_2024-',
    data_type='clear'
)

atm_cloudy_valid, sfc_cloudy_valid, hydro_cloudy_valid, out_cloudy_valid = load_day_data_bath(
    sim_dir='/root/autodl-tmp/cloudy2024',
    nc_dir='/root/autodl-tmp/cloudy_sample',
    prefix_sim='cloudy_2024_',
    prefix_nc='ifs_cloudy_2024-',
    data_type='cloudy'
)

# Set TCLW and TCRW to 0 for clear-sky data (physical consistency)
sfc_clear_train[:, -2:] = 0 
sfc_clear_valid[:, -2:] = 0

print(f"\nData Statistics:")
print(f"Train Clear: atm={atm_clear_train.shape}, sfc={sfc_clear_train.shape}, out={out_clear_train.shape}")
print(f"Train Cloudy: atm={atm_cloudy_train.shape}, sfc={sfc_cloudy_train.shape}, hydro={hydro_cloudy_train.shape}, out={out_cloudy_train.shape}")
print(f"Valid Clear: atm={atm_clear_valid.shape}, sfc={sfc_clear_valid.shape}, out={out_clear_valid.shape}")
print(f"Valid Cloudy: atm={atm_cloudy_valid.shape}, sfc={sfc_cloudy_valid.shape}, hydro={hydro_cloudy_valid.shape}, out={out_cloudy_valid.shape}")


# ============================================================================
# 2. Load Jacobian Matrices
# ============================================================================

print("\n2. Loading Jacobian Matrices...")

# --- Training Set Jacobians ---
print("\nLoading Training Jacobians...")
jac_atm_clear_train, jac_sfc_clear_train, _ = load_J_data_batch(
    jac_dir='/root/autodl-tmp/Jmatrix/clear',
    data_type='clear'
)

jac_atm_cloudy_train, jac_sfc_cloudy_train, jac_cloud_cloudy_train = load_J_data_batch(
    jac_dir='/root/autodl-tmp/Jmatrix/cloudy',
    data_type='cloudy'
)

# --- Validation Set Jacobians ---
print("\nLoading Validation Jacobians...")
jac_atm_clear_valid, jac_sfc_clear_valid, _ = load_J_data_batch(
    jac_dir='/root/autodl-tmp/Jmatrix/valid_clear',
    data_type='clear'
)

jac_atm_cloudy_valid, jac_sfc_cloudy_valid, jac_cloud_cloudy_valid = load_J_data_batch(
    jac_dir='/root/autodl-tmp/Jmatrix/valid_cloudy',
    data_type='cloudy'
)

print(f"\nJacobian Data Statistics:")
print(f"Train Clear Jacobian: atm={jac_atm_clear_train.shape}, sfc={jac_sfc_clear_train.shape}")
print(f"Train Cloudy Jacobian: atm={jac_atm_cloudy_train.shape}, sfc={jac_sfc_cloudy_train.shape}, cloud={jac_cloud_cloudy_train.shape}")
print(f"Valid Clear Jacobian: atm={jac_atm_clear_valid.shape}, sfc={jac_sfc_clear_valid.shape}")
print(f"Valid Cloudy Jacobian: atm={jac_atm_cloudy_valid.shape}, sfc={jac_sfc_cloudy_valid.shape}, cloud={jac_cloud_cloudy_valid.shape}")


# ============================================================================
# 3. Standardization (Normalization)
# ============================================================================

print("\n3. Standardizing Input Features and Output (Applying Min-Max Scaling)...")

scaler_dir = './scalers'
os.makedirs(scaler_dir, exist_ok=True)

# Load pre-fitted scalers (Global statistics)
scaler_atm_list = joblib.load(f'{scaler_dir}/scaler_atm_global.pkl')
scaler_sfc = joblib.load(f'{scaler_dir}/scaler_sfc_global.pkl')
scaler_hydro_list = joblib.load(f'{scaler_dir}/scaler_hydro_cloudy.pkl')
scaler_y = joblib.load(f'{scaler_dir}/scaler_y.pkl')

print("Successfully loaded pre-fitted scalers")

def normalize_atm(atm_data, scaler_list):
    """Standardize atmospheric data using per-level scalers."""
    atm_scaled = np.zeros_like(atm_data)
    for k in range(atm_data.shape[1]): # Loop over 37 levels
        atm_scaled[:, k, :] = scaler_list[k].transform(atm_data[:, k, :])
    return atm_scaled

def normalize_hydro(hydro_data, scaler_list):
    """Standardize hydrometeor data using per-level scalers."""
    hydro_scaled = np.zeros_like(hydro_data)
    for k in range(hydro_data.shape[1]): # Loop over 36 layers
        hydro_scaled[:, k, :] = scaler_list[k].transform(hydro_data[:, k, :])
    return hydro_scaled

# Apply Standardization
# --- Training Set ---
atm_clear_train_scaled = normalize_atm(atm_clear_train, scaler_atm_list)
sfc_clear_train_scaled = scaler_sfc.transform(sfc_clear_train)
out_clear_train_scaled = scaler_y.transform(out_clear_train)

atm_cloudy_train_scaled = normalize_atm(atm_cloudy_train, scaler_atm_list)
sfc_cloudy_train_scaled = scaler_sfc.transform(sfc_cloudy_train)
hydro_cloudy_train_scaled = normalize_hydro(hydro_cloudy_train, scaler_hydro_list)
out_cloudy_train_scaled = scaler_y.transform(out_cloudy_train)

# --- Validation Set ---
atm_clear_valid_scaled = normalize_atm(atm_clear_valid, scaler_atm_list)
sfc_clear_valid_scaled = scaler_sfc.transform(sfc_clear_valid)
out_clear_valid_scaled = scaler_y.transform(out_clear_valid)

atm_cloudy_valid_scaled = normalize_atm(atm_cloudy_valid, scaler_atm_list)
sfc_cloudy_valid_scaled = scaler_sfc.transform(sfc_cloudy_valid)
hydro_cloudy_valid_scaled = normalize_hydro(hydro_cloudy_valid, scaler_hydro_list)
out_cloudy_valid_scaled = scaler_y.transform(out_cloudy_valid)


# ============================================================================
# 4. Compute Jacobian Statistics (for Denormalization during training)
# ============================================================================

print("\n4. Computing Jacobian Statistics (Min-Max values for Denormalization)")

def compute_jacobian_stats(jac_atm_clear, jac_atm_cloudy, jac_sfc_clear, jac_sfc_cloudy, jac_cloud_cloudy):
    """
    Compute statistics (min/max) for Jacobian matrices.
    These stats are used to denormalize Jacobian predictions during training.
    """
    jac_stats = {}

    # Atmospheric Jacobian Stats (37, 22, 2)
    print("Computing Atmospheric Jacobian stats...")
    all_jac_atm = np.concatenate([jac_atm_clear, jac_atm_cloudy], axis=0)
    jac_stats['atm_min'] = np.min(all_jac_atm, axis=0) # Shape: (37, 22, 2)
    jac_stats['atm_max'] = np.max(all_jac_atm, axis=0)

    # Surface Jacobian Stats (22, 3)
    print("Computing Surface Jacobian stats...")
    all_jac_sfc = np.concatenate([jac_sfc_clear, jac_sfc_cloudy], axis=0)
    jac_stats['sfc_min'] = np.min(all_jac_sfc, axis=0) # Shape: (22, 3)
    jac_stats['sfc_max'] = np.max(all_jac_sfc, axis=0)

    # Cloud Jacobian Stats (36, 22, 4)
    print("Computing Cloud Jacobian stats...")
    jac_stats['cloud_min'] = np.min(jac_cloud_cloudy, axis=0) # Shape: (36, 22, 4)
    jac_stats['cloud_max'] = np.max(jac_cloud_cloudy, axis=0)

    return jac_stats

# Calculate stats based on Training Set
print("Calculating Jacobian stats from Training Set...")
jacobian_stats = compute_jacobian_stats(
    jac_atm_clear_train, jac_atm_cloudy_train,
    jac_sfc_clear_train, jac_sfc_cloudy_train,
    jac_cloud_cloudy_train
)

# Save Jacobian stats
joblib.dump(jacobian_stats, f'{scaler_dir}/jacobian_stats.pkl')
print("Jacobian statistics saved successfully")


# ============================================================================
# 5. Save to HDF5
# ============================================================================

print("\n5. Saving Processed Data to HDF5 File...")

h5_path = './scalers/radiation_data_jacobian.h5'

with h5py.File(h5_path, 'w') as h5_file:
    # Create groups
    train_group = h5_file.create_group('train')
    valid_group = h5_file.create_group('valid')
    stats_group = h5_file.create_group('jacobian_stats')
    
    # --- Training Set ---
    print("Saving Training Set...")
    
    # Clear Sky
    clear_train_group = train_group.create_group('clear')
    clear_train_group.create_dataset('atm', data=atm_clear_train_scaled, compression='gzip')
    clear_train_group.create_dataset('sfc', data=sfc_clear_train_scaled, compression='gzip')
    clear_train_group.create_dataset('out', data=out_clear_train_scaled, compression='gzip')
    clear_train_group.create_dataset('jac_atm', data=jac_atm_clear_train, compression='gzip')
    clear_train_group.create_dataset('jac_sfc', data=jac_sfc_clear_train, compression='gzip')
    
    # cloudy sky
    cloudy_train_group = train_group.create_group('cloudy')
    cloudy_train_group.create_dataset('atm', data=atm_cloudy_train_scaled, compression='gzip')
    cloudy_train_group.create_dataset('sfc', data=sfc_cloudy_train_scaled, compression='gzip')
    cloudy_train_group.create_dataset('hydro', data=hydro_cloudy_train_scaled, compression='gzip')
    cloudy_train_group.create_dataset('out', data=out_cloudy_train_scaled, compression='gzip')
    cloudy_train_group.create_dataset('jac_atm', data=jac_atm_cloudy_train, compression='gzip')
    cloudy_train_group.create_dataset('jac_sfc', data=jac_sfc_cloudy_train, compression='gzip')
    cloudy_train_group.create_dataset('jac_cloud', data=jac_cloud_cloudy_train, compression='gzip')
    
    # --- Validation Set ---
    print("Saving Validation Set...")
    
    # Clear Sky
    clear_valid_group = valid_group.create_group('clear')
    clear_valid_group.create_dataset('atm', data=atm_clear_valid_scaled, compression='gzip')
    clear_valid_group.create_dataset('sfc', data=sfc_clear_valid_scaled, compression='gzip')
    clear_valid_group.create_dataset('out', data=out_clear_valid_scaled, compression='gzip')
    clear_valid_group.create_dataset('jac_atm', data=jac_atm_clear_valid, compression='gzip')
    clear_valid_group.create_dataset('jac_sfc', data=jac_sfc_clear_valid, compression='gzip')
    
    # cloudy sky
    cloudy_valid_group = valid_group.create_group('cloudy')
    cloudy_valid_group.create_dataset('atm', data=atm_cloudy_valid_scaled, compression='gzip')
    cloudy_valid_group.create_dataset('sfc', data=sfc_cloudy_valid_scaled, compression='gzip')
    cloudy_valid_group.create_dataset('hydro', data=hydro_cloudy_valid_scaled, compression='gzip')
    cloudy_valid_group.create_dataset('out', data=out_cloudy_valid_scaled, compression='gzip')
    cloudy_valid_group.create_dataset('jac_atm', data=jac_atm_cloudy_valid, compression='gzip')
    cloudy_valid_group.create_dataset('jac_sfc', data=jac_sfc_cloudy_valid, compression='gzip')
    cloudy_valid_group.create_dataset('jac_cloud', data=jac_cloud_cloudy_valid, compression='gzip')
    
    # Save Jacobian statistical information
    print("Save Jacobian statistical information...")
    stats_group.create_dataset('atm_min', data=jacobian_stats['atm_min'])
    stats_group.create_dataset('atm_max', data=jacobian_stats['atm_max'])
    stats_group.create_dataset('sfc_min', data=jacobian_stats['sfc_min'])
    stats_group.create_dataset('sfc_max', data=jacobian_stats['sfc_max'])
    stats_group.create_dataset('cloud_min', data=jacobian_stats['cloud_min'])
    stats_group.create_dataset('cloud_max', data=jacobian_stats['cloud_max'])
    
    # Save the shape information of the data
    shapes_group = h5_file.create_group('shapes')
    shapes_group.create_dataset('atm_shape', data=np.array([37, 3]))
    shapes_group.create_dataset('sfc_shape', data=np.array([5]))
    shapes_group.create_dataset('hydro_shape', data=np.array([36, 4]))
    shapes_group.create_dataset('out_shape', data=np.array([22]))
    shapes_group.create_dataset('jac_atm_shape', data=np.array([37, 22, 2]))
    shapes_group.create_dataset('jac_sfc_shape', data=np.array([22, 3]))
    shapes_group.create_dataset('jac_cloud_shape', data=np.array([36, 22, 4]))

print(f"\n✅ Data saved successfully: {h5_path}")
