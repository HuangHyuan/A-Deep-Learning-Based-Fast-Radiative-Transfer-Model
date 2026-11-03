"""Shared training and Jacobian helpers."""

import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def forward_model(model, batch):
    atm, sfc, hydro = batch["atm_input"], batch["sfc_input"], batch["hydro_input"]
    cloudy = batch["is_cloudy"] > 0.5
    prediction = torch.empty((atm.shape[0], model.n_channels), device=atm.device, dtype=atm.dtype)
    clear_index = torch.where(~cloudy)[0]
    cloudy_index = torch.where(cloudy)[0]
    if clear_index.numel():
        prediction[clear_index] = model(atm[clear_index], sfc[clear_index])
    if cloudy_index.numel():
        prediction[cloudy_index] = model(atm[cloudy_index], sfc[cloudy_index], hydro[cloudy_index])
    return prediction


def analytical_jacobians(model, atm, sfc, hydro=None):
    model.eval()
    n_samples, n_layers, _ = atm.shape
    n_channels = model.n_channels
    jac_atm = np.zeros((n_samples, n_layers, n_channels, 2), dtype=np.float32)
    jac_sfc = np.zeros((n_samples, n_channels, 3), dtype=np.float32)
    jac_cloud = np.zeros((n_samples, n_layers, n_channels, 4), dtype=np.float32)
    for sample_idx in range(n_samples):
        sample_atm = atm[sample_idx:sample_idx + 1].detach().requires_grad_(True)
        sample_sfc = sfc[sample_idx:sample_idx + 1].detach().requires_grad_(True)
        if hydro is None:
            output = model(sample_atm, sample_sfc)
            inputs = (sample_atm, sample_sfc)
        else:
            sample_hydro = hydro[sample_idx:sample_idx + 1].detach().requires_grad_(True)
            output = model(sample_atm, sample_sfc, sample_hydro)
            inputs = (sample_atm, sample_sfc, sample_hydro)
        grad_atm, grad_sfc, grad_hydro = [], [], []
        for channel in range(n_channels):
            grads = torch.autograd.grad(output[0, channel], inputs,
                                        retain_graph=channel < n_channels - 1)
            grad_atm.append(grads[0][0].detach().cpu().numpy())
            grad_sfc.append(grads[1][0, :3].detach().cpu().numpy())
            if hydro is not None:
                grad_hydro.append(grads[2][0].detach().cpu().numpy())
        grad_atm_array = np.stack(grad_atm, axis=0)
        jac_atm[sample_idx, :, :, :] = np.transpose(grad_atm_array, (1, 0, 2))[:, :, :2]
        jac_sfc[sample_idx] = np.stack(grad_sfc, axis=1).reshape(n_channels, 3)
        if hydro is not None:
            grad_hydro_array = np.stack(grad_hydro, axis=0)
            jac_cloud[sample_idx] = np.transpose(grad_hydro_array, (1, 0, 2))
    return jac_atm, jac_sfc, jac_cloud


def normalized_jacobian_loss(model_jac, target_jac):
    if model_jac.numel() == 0 or target_jac.numel() == 0:
        return model_jac.sum() * 0.0
    reduce_dims = tuple(range(0, target_jac.ndim - 2))
    scale = target_jac.detach().pow(2).mean(dim=reduce_dims).sqrt().clamp_min(1e-6)
    while scale.ndim < target_jac.ndim:
        scale = scale.unsqueeze(0)
    return F.smooth_l1_loss(model_jac / scale, target_jac / scale)


def batch_jacobian_loss(model, batch, device, max_samples=4):
    n_use = min(max_samples, batch["atm_input"].shape[0])
    atm = batch["atm_input"][:n_use].to(device)
    sfc = batch["sfc_input"][:n_use].to(device)
    hydro = batch["hydro_input"][:n_use].to(device)
    cloudy = batch["is_cloudy"][:n_use].to(device) > 0.5
    loss = atm.sum() * 0.0
    terms = 0
    if (~cloudy).any():
        idx = torch.where(~cloudy)[0]
        model_atm, model_sfc, _ = analytical_jacobians(model, atm[idx], sfc[idx])
        loss += normalized_jacobian_loss(torch.from_numpy(model_atm).to(device),
                                         batch["jac_atm"][:n_use][idx].to(device)[:, :, :, :2].float())
        loss += normalized_jacobian_loss(torch.from_numpy(model_sfc).to(device),
                                         batch["jac_sfc"][:n_use][idx].to(device)[:, :, :3].float())
        terms += 2
    if cloudy.any():
        idx = torch.where(cloudy)[0]
        model_atm, model_sfc, model_cloud = analytical_jacobians(model, atm[idx], sfc[idx], hydro[idx])
        loss += normalized_jacobian_loss(torch.from_numpy(model_atm).to(device),
                                         batch["jac_atm"][:n_use][idx].to(device)[:, :, :, :2].float())
        loss += normalized_jacobian_loss(torch.from_numpy(model_sfc).to(device),
                                         batch["jac_sfc"][:n_use][idx].to(device)[:, :, :3].float())
        loss += normalized_jacobian_loss(torch.from_numpy(model_cloud).to(device),
                                         batch["jac_cloud"][:n_use][idx].to(device).float())
        terms += 3
    return loss / max(terms, 1)


def graph_jacobians(model, atm, sfc, hydro=None):
    """Differentiable Jacobians used by the optional consistency loss."""

    atm_jac, sfc_jac, hydro_jac = [], [], []
    for sample_idx in range(atm.shape[0]):
        sample_atm = atm[sample_idx:sample_idx + 1].detach().requires_grad_(True)
        sample_sfc = sfc[sample_idx:sample_idx + 1].detach().requires_grad_(True)
        if hydro is None:
            output = model(sample_atm, sample_sfc)
            inputs = (sample_atm, sample_sfc)
        else:
            sample_hydro = hydro[sample_idx:sample_idx + 1].detach().requires_grad_(True)
            output = model(sample_atm, sample_sfc, sample_hydro)
            inputs = (sample_atm, sample_sfc, sample_hydro)
        atm_terms, sfc_terms, hydro_terms = [], [], []
        for channel in range(model.n_channels):
            grads = torch.autograd.grad(output[0, channel], inputs, retain_graph=True,
                                        create_graph=True)
            atm_terms.append(grads[0][0, :, :2])
            sfc_terms.append(grads[1][0, :3])
            if hydro is not None:
                hydro_terms.append(grads[2][0])
        atm_jac.append(torch.stack(atm_terms, dim=0).permute(1, 0, 2))
        sfc_jac.append(torch.stack(sfc_terms, dim=0))
        if hydro is not None:
            hydro_jac.append(torch.stack(hydro_terms, dim=0).permute(1, 0, 2))
    if hydro is None:
        return torch.stack(atm_jac), torch.stack(sfc_jac), None
    return torch.stack(atm_jac), torch.stack(sfc_jac), torch.stack(hydro_jac)


def batch_jacobian_loss(model, batch, device, max_samples=4):
    """Redefinition that keeps the Jacobian graph for backpropagation."""

    n_use = min(max_samples, batch["atm_input"].shape[0])
    atm = batch["atm_input"][:n_use].to(device)
    sfc = batch["sfc_input"][:n_use].to(device)
    hydro = batch["hydro_input"][:n_use].to(device)
    cloudy = batch["is_cloudy"][:n_use].to(device) > 0.5
    loss = atm.sum() * 0.0
    terms = 0
    if (~cloudy).any():
        idx = torch.where(~cloudy)[0]
        pred_atm, pred_sfc, _ = graph_jacobians(model, atm[idx], sfc[idx])
        loss += normalized_jacobian_loss(
            pred_atm, batch["jac_atm"][:n_use][idx].to(device)[:, :, :, :2].float()
        )
        loss += normalized_jacobian_loss(
            pred_sfc, batch["jac_sfc"][:n_use][idx].to(device)[:, :, :3].float()
        )
        terms += 2
    if cloudy.any():
        idx = torch.where(cloudy)[0]
        pred_atm, pred_sfc, pred_cloud = graph_jacobians(
            model, atm[idx], sfc[idx], hydro[idx]
        )
        loss += normalized_jacobian_loss(
            pred_atm, batch["jac_atm"][:n_use][idx].to(device)[:, :, :, :2].float()
        )
        loss += normalized_jacobian_loss(
            pred_sfc, batch["jac_sfc"][:n_use][idx].to(device)[:, :, :3].float()
        )
        loss += normalized_jacobian_loss(
            pred_cloud, batch["jac_cloud"][:n_use][idx].to(device).float()
        )
        terms += 3
    return loss / max(terms, 1)
