# coding=utf-8
    # Copyright (C) 2025  Jean-Daniel PASCAL PRIETO

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""All functions related to loss computation and optimization."""

import logging

import numpy as np
import torch
import torch.optim as optim
from lightning import Fabric
import logging

from models import utils as mutils
from sde_lib import SDE, VESDE, VPSDE


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == "Adam":
        optimizer = optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(
        optimizer,
        params,
        step,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
    ):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_sde_loss_fn(
    sde, train, reduce_mean=True, continuous=True, loss_type="score_matching", eps=1e-5
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      loss_type: string, the type of loss function to use. Options are:
        - "score_matching": Use the score matching loss.
        - "denoiser": Use the denoising loss.
        - "data_prediction": Use the data prediction loss. DDPM type, recommended for Schrodinger bridges

    Returns:
      A loss function.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        x, y = batch
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps   
        z = torch.randn_like(x) 
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z 

        score = score_fn(perturbed_data, t, y)
                
        if loss_type == "score_matching":
            losses = torch.square(score * std[:, None, None, None] + z) # Eq. (7)

            # Sum over spatial dimensions and channels and mean over batch
            # losses =  0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) #* std[:, None, None, None].pow(2)
        elif loss_type == "denoiser":
            D = score * std[:, None, None, None].pow(2) + perturbed_data # equivalent to Eq. (10)
            losses = torch.square(torch.abs(D - mean)) # Eq. (8)
            # Sum over spatial dimensions and channels and mean over batch
            losses = 0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
        elif loss_type == "data_prediction":
            logging.info("Using data prediction loss")
            # B, C, T, Ch = x.shape

            losses = torch.square(score - x)
            # losses = 0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) #* std[:, None, None, None].pow(2)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


        if train:
            # IDEA: Add noise to the loss for training
            # noise = torch.randn_like(losses)
            # losses = losses + eps * noise
            loss = torch.mean(losses)
        else:
            return losses

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],))
        sigmas = smld_sigma_array[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        target = -noise / (sigmas**2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas**2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        noise = torch.randn_like(batch)
        perturbed_data = (
            sqrt_alphas_cumprod[labels, None, None, None] * batch
            + sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        )
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(
    sde: SDE,
    fabric: Fabric,
    train: bool,
    optimize_fn=None,
    reduce_mean=False,
    continuous=True,
    loss_type="score_matching",
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      loss_type: string, the type of loss function to use. Options are:
        - "score_matching": Use the score matching loss.
        - "denoiser": Use the denoising loss.
        - "data_prediction": Use the data prediction loss. DDPM type, recommended for Schrodinger bridges

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(
            sde,
            train,
            reduce_mean=reduce_mean,
            continuous=True,
            loss_type=loss_type,
        )
    else:
        assert not loss_type, (
            "Likelihood weighting is not supported for original SMLD/DDPM training."
        )
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(
                f"Discrete training for {sde.__class__.__name__} is not recommended."
            )

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model: torch.nn.Module = state["model"]

        if train:
            optimizer = state["optimizer"]
            
            logging.debug(" -> running optimizer.zero_grad()")
            optimizer.zero_grad()
        
            loss = loss_fn(model, batch)
            
            logging.debug(" -> running loss.backward()")
            fabric.backward(loss)
            
            logging.debug(
                " -> running optimize_fn(optimizer, model.parameters(), step=state['step'])"
            )
            optimize_fn(optimizer, model.parameters(), step=state["step"])
            
            logging.debug(" -> incrementing step")
            state["step"] += 1
            logging.debug(" -> running state['ema'].update(model.parameters())")
            state["ema"].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state["ema"]
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn
