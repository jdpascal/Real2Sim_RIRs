# coding=utf-8
# pylint: skip-file

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

"""Training and evaluation for score-based generative models."""

import gc
import glob
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from absl import flags
from lightning import Fabric
from ml_collections import ConfigDict
import torch
from torch.utils.tensorboard import SummaryWriter

import datasets
import likelihood
import losses
import sampling
import sde_lib

# Importer tous les modèles pour les enregistrer
from models import ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, save_checkpoint

FLAGS = flags.FLAGS


def train(config: ConfigDict, workdir: Path, fabric: Fabric):
    """
    Lance la phase d'entraînement.

    Args:
        config: Objet de configuration.
        workdir: Répertoire de travail pour sauvegardes et logs TensorBoard.
    """

    # Initialisation du modèle.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    score_model, optimizer = fabric.setup(score_model, optimizer)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    logging.info("Creating checkpoints dirs...")
    # Creation of the folders for the checkpoints.
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_meta_dir = workdir / "checkpoints-meta" / "checkpoint.pth"
    checkpoint_meta_dir.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Restore checkpoints...")
    # Continue the training from the latest checkpoint if it exists.
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state["step"])

    logging.info("Building dataloaders...")
    # Construction of the DataLoaders (version PyTorch).
    train_loader, eval_loader, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
    )

    train_loader = fabric.setup_dataloaders(train_loader)
    eval_loader = fabric.setup_dataloaders(eval_loader)
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    logging.info("Creating normalization and inverse functions...")
    # Creation of the normalization and inverse functions.
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Configuration of the SDE.
    sde_name = config.training.sde.lower()
    if sde_name == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif sde_name == "ouvesde":
        sde = sde_lib.OUVESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    elif sde_name == "sbvesde":
        sde = sde_lib.SBVESDE(
            k=config.model.k,
            c=config.model.c,
            N=config.model.num_scales
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} inconnu.")

    # Set up the training step function.
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    loss_type = config.training.loss_type
    train_step_fn = losses.get_step_fn(
        sde,
        fabric,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        loss_type=loss_type,
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        fabric,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        loss_type=loss_type,
    )

    num_train_steps = config.training.n_iters
    logging.info("Début de la boucle d'entraînement à l'étape %d.", initial_step)

    # Log all configuration in tensorboard
    if fabric.is_global_zero:
        fabric.logger.log_hyperparams(config.to_dict())

    for step in range(initial_step, num_train_steps + 1):
        logging.info("Loop start %d", step)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # If the dataset provides both 'perfect_rir' and 'real_rir', we use them as a tuple for training.
        if "perfect_rir" in batch and "real_rir" in batch:
            perfect_rir = batch["perfect_rir"]

            # Si nécessaire, ajuster l'ordre des dimensions (ex. HWC -> CHW)
            if (
                perfect_rir.ndim == 4
                and perfect_rir.shape[-1] != config.data.num_channels
            ):
                perfect_rir = perfect_rir.permute(0, 2, 3, 1)
            real_rir = batch["real_rir"]

            if real_rir.ndim == 4 and real_rir.shape[-1] != config.data.num_channels:
                real_rir = real_rir.permute(0, 2, 3, 1)
            current_batch = (perfect_rir, real_rir)
            del perfect_rir, real_rir
        else:
            raise ValueError("Le batch doit contenir à la fois 'perfect_rir' et 'real_rir' pour l'entraînement.")

        # logging.info(f"memory summary before training: {torch.cuda.memory_summary()}")

        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        # Execute a training step
        loss = train_step_fn(state, current_batch)
        # del current_batch

        if step % config.training.log_freq == 0:
            loss = fabric.all_gather(loss).mean()
            # Gather loss from all processes, log value only on process with rank 0
            if fabric.is_global_zero:
                logging.info("étape: %d, loss entraînement: %.5e", step, loss.item())
                fabric.log("training_loss", loss.item(), step)

        # Save checkpoints for preemption in cloud computing environments.
        # Run only on process with rank 0
        if (
            step != 0
            and step % config.training.snapshot_freq_for_preemption == 0
            and fabric.global_rank == 0
        ):
            save_checkpoint(checkpoint_meta_dir, state)

        # Évaluation périodique sur le jeu de validation.
        if step % config.training.eval_freq == 0:
            try:
                eval_batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_loader)
                eval_batch = next(eval_iter)
            eval_img_perfect = eval_batch["perfect_rir"].to(fabric.device).detach()
            eval_img_real = eval_batch["real_rir"].to(fabric.device).detach()
            if (
                eval_img_perfect.ndim == 4
                and eval_img_perfect.shape[-1] != config.data.num_channels
            ):
                eval_img_perfect = eval_img_perfect.permute(0, 2, 3, 1)
                eval_img_real = eval_img_real.permute(0, 2, 3, 1)

            eval_img_batch = (eval_img_perfect, eval_img_real)
            eval_loss = eval_step_fn(state, eval_img_batch)
            eval_loss = fabric.all_gather(eval_loss).mean()
            # Run logging only on process with rank 0
            if fabric.is_global_zero:
                logging.info("étape: %d, loss évaluation: %.5e", step, eval_loss.item())
                fabric.log("eval_loss", eval_loss.item(), step)
            del eval_img_batch, eval_img_perfect, eval_img_real, eval_batch

        # Sauvegarde d'un checkpoint complet et génération d'échantillons.
        # Run this only on process with rank 0
        if (
            (step != 0 and step % config.training.snapshot_freq == 0)
            or (step == num_train_steps)
            and fabric.is_global_zero
        ):
            # Sauvegarde du checkpoint.
            save_step = step // config.training.snapshot_freq
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth")
            save_checkpoint(ckpt_path, state)

            # Génération d'échantillons.
            if config.training.snapshot_sampling:
                sampling_shape = (
                        config.training.batch_size,
                        config.data.num_channels,
                        config.data.rir_samples_count,
                        config.data.channels,
                    )
                optimizer.zero_grad()
                with torch.no_grad():
                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sampling_fn = sampling.get_sampling_fn(
                        config,
                        sde,
                        sampling_shape,
                        inverse_scaler,
                        sampling_eps,
                        y=current_batch[1].detach(),
                    )
                    samples, n = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                for index, sample in enumerate(samples):
                    # Sélection du premier exemple du batch pour la comparaison
                    perfect_rir_sample = (
                        current_batch[0][0].detach().cpu().numpy()
                    )  # forme: (epaisseur=1, longueur, canaux)
                    generated_sample = (
                        sample.detach().cpu().numpy()
                    )  # forme: (epaisseur=1, longueur, canaux)
                    # Suppression de la dimension "épaisseur" (qui vaut 1)
                    perfect_rir_sample = np.squeeze(
                        perfect_rir_sample, axis=0
                    )  # devient (longueur, canaux)
                    generated_sample = np.squeeze(
                        generated_sample, axis=0
                    )  # devient (longueur, canaux)
                    real_rir_sample = (
                        current_batch[1][0].detach().cpu().numpy()
                    )
                    # Suppression de la dimension "épaisseur" (qui vaut 1)
                    real_rir_sample = np.squeeze(
                        real_rir_sample, axis=0
                    )  # devient (longueur, canaux)

                    # On trace jusqu'à 32 canaux (ou le nombre maximum de canaux disponibles)
                    plt.ioff()
                    num_channels = min(32, perfect_rir_sample.shape[1])
                    fig, axes = plt.subplots(
                        nrows=num_channels,
                        ncols=1,
                        sharex=True,
                        figsize=(12,24),
                        layout="constrained",
                    )
                    for c in range(num_channels):
                        # Puisque ce sont des signaux 1D, on les trace directement.
                        signal_sample = generated_sample[:, c]
                        signal_perfect = perfect_rir_sample[:, c] 
                        signal_real = real_rir_sample[:, c]
                        axes[c].plot(signal_sample, label="Generated RIR")
                        axes[c].plot(signal_perfect, label="Perfect RIR")
                        axes[c].plot(signal_real, label="Real RIR",linestyle = '-', linewidth=0.5)
                        axes[c].set_ylim(bottom=-1.5, top=1.5)

                    axes[-1].set_xlabel("Time")
                    axes[-1].legend()
                    fig.suptitle("Signal per channels")
                    # fig.tight_layout()
                    fig.subplots_adjust(hspace=0)
                    fabric.logger.experiment.add_figure(f"sample_at_step_{step}", fig, index)
                    plt.close()
                    del fig, axes, signal_sample, signal_perfect, signal_real, perfect_rir_sample, generated_sample, real_rir_sample
                    torch.cuda.empty_cache()


def evaluate(config, workdir, eval_folder="eval", fabric=None):
    """
    Évalue les modèles entraînés.
    Args:
        config: Objet de configuration.
        workdir: Répertoire de travail contenant les checkpoints.
        eval_folder: Sous-dossier pour stocker les résultats d'évaluation.
        fabric: Fabric pour la gestion des ressources distribuées.
    """
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Chargement des DataLoaders en mode évaluation.
    _, eval_loader, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        evaluation=True,
    )

    eval_loader = fabric.setup_dataloaders(eval_loader)

    # scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialisation du modèle, de l'optimizer et de l'EMA.
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    score_model, optimizer = fabric.setup(score_model, optimizer)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints_SB")

    # Configuration de la SDE.
    sde_name = config.training.sde.lower()
    if sde_name == "ouvesde":
        sde = sde_lib.OUVESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    elif sde_name == "sbvesde":
        sde = sde_lib.SBVESDE(
            k=config.model.k,
            c=config.model.c,
            N=config.model.num_scales
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} inconnu.")

    # Création de la fonction d'évaluation de la loss si activée.
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        loss_type = config.training.loss_type
        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(
            sde,
            fabric,
            train=False,
            optimize_fn=optimize_fn,
            reduce_mean=reduce_mean,
            continuous=continuous,
            loss_type=loss_type,
        )

    sampling_shape = (
        config.eval.batch_size,
        config.data.num_channels,
        config.data.rir_samples_count,
        config.data.channels,
    )

    begin_ckpt = config.eval.begin_ckpt
    logging.info("Début de l'évaluation à partir du checkpoint %d.", begin_ckpt)

    # Log all configuration in tensorboard
    if fabric.is_global_zero:
        fabric.logger.log_hyperparams(config.to_dict())

    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
        waiting_message_printed = False
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("En attente du checkpoint_%d", ckpt)
                waiting_message_printed = True
            time.sleep(60)

        # Tentative de chargement du checkpoint.
        try:
            state = restore_checkpoint(ckpt_filename, state, config.device)
        except Exception:
            logging.warning(
                "Problème lors du chargement du checkpoint, nouvelle tentative dans 60s..."
            )
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_filename, state, config.device)
            except Exception:
                time.sleep(120)
                state = restore_checkpoint(ckpt_filename, state, config.device)

        logging.info("Évaluation du checkpoint %d", ckpt)

        ema.copy_to(score_model.parameters())
        logging.info(sum(p.numel() for p in score_model.parameters() if p.requires_grad))

        # On garde une copie de la première rir
        # rir_chunks_count = config.data.total_rir_samples_count // config.data.rir_samples_count 

        # Évaluation de la loss sur l'ensemble du dataset d'évaluation.
        if config.eval.enable_loss:
            step = state["step"]
            all_losses = [ ]

            # for i in range(int(rir_chunks_count)):
            #     all_losses[i] = []
            for i, batch in enumerate(eval_loader):
                logging.info("Évaluation de la loss sur le batch %d", i)
                perfect_rir, real_rir, verite, ordre, geometry = batch["perfect_rir"], batch["real_rir"], batch['verite'], batch['ordre'], batch['geometry']

                if real_rir.ndim == 4 and real_rir.shape[-1] != config.data.num_channels:
                    real_rir = real_rir.permute(0, 2, 3, 1)

                if perfect_rir.ndim == 4 and perfect_rir.shape[-1] != config.data.num_channels:
                    perfect_rir = perfect_rir.permute(0, 2, 3, 1)

                eval_batch = perfect_rir, real_rir
                loss_eval = eval_step(state, eval_batch)
                del eval_batch

                # chunk_index = chunk_index.cpu().numpy()
                loss_eval = loss_eval.cpu().numpy()

                # for chunk in range(len(chunk_index)) :
                #     all_losses[chunk_index[chunk]].append(loss_eval[chunk])
                # all_losses[chunk_index].append(loss_eval.item())
                all_losses.append(loss_eval.item())

                # if (i + 1) * config.eval.batch_size <= rir_chunks_count :
                if True :
                    # On remplit le tableau avec les valeurs de la batch
                    sampling_fn = sampling.get_sampling_fn(
                            config,
                            sde,
                            sampling_shape,
                            inverse_scaler,
                            sampling_eps,
                            y=real_rir.detach(),
                    )
                    # Génération d'échantillons.
                    this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
                    os.makedirs(this_sample_dir, exist_ok=True)
                    samples, n = sampling_fn(score_model)
                    logging.info("samples shape: %s", len(samples))
                    logging.info("samples shape: %s", samples[0].shape)

                    if config.eval.enable_sampling:
                        for index, sample in enumerate(samples[-1]): #[-1]
                            # Sauvegarde des échantillons générés.
                            sample_filepath = os.path.join(
                                this_sample_dir, f"sample_{index}_{i}.npz" #_chunk{i}
                            )
                            # if index == 199:
                            for k, v in geometry.items():
                                # print(f"geometry {k} shape: {len(v)}, dtype: {v.dtype}, device: {v.device}")
                                v = torch.tensor(v)  # Convertir en tensor PyTorch
                                geometry[k] = v.to("cpu").numpy()
                            with open(sample_filepath, "wb") as f:
                                np.savez_compressed(
                                    f,
                                    generated_rir=sample.detach().cpu().numpy(),
                                    perfect_rir=perfect_rir.detach().cpu().numpy(),
                                    real_rir=real_rir.detach().cpu().numpy(),
                                    verite=verite.cpu().numpy(),
                                    ordre=ordre.cpu().numpy(),
                                    geometry=geometry,
                                )

                    # Sélection du premier exemple du batch pour la comparaison
                    perfect_rir_sample = (
                            perfect_rir[0].detach().cpu().numpy()
                        )  # forme: (epaisseur=1, longueur, canaux)
                                        # Suppression de la dimension "épaisseur" (qui vaut 1)
                    perfect_rir_sample = np.squeeze(
                            perfect_rir_sample, axis=0
                        )  # devient (longueur, canaux)
                    real_rir_sample = (
                            real_rir[0].detach().cpu().numpy()
                        )
                        # Suppression de la dimension "épaisseur" (qui vaut 1)
                    real_rir_sample = np.squeeze(
                            real_rir_sample, axis=0
                        )  # devient (longueur, canaux)
                    generated_sample = (
                            samples[-1].detach().cpu().numpy()   #[-1]
                        )  # forme: (epaisseur=1, longueur, canaux)
                    del samples
                    generated_sample = np.squeeze(
                            generated_sample, axis=0
                        )  # devient (longueur, canaux)

                    # if (i + 1) * config.eval.batch_size == rir_chunks_count:
                    if True :
                        # On trace jusqu'à 32 canaux (ou le nombre maximum de canaux disponibles)
                        plt.ioff()
                        num_channels = min(32, perfect_rir_sample.shape[1])
                        fig, axes = plt.subplots(
                            nrows=num_channels,
                            ncols=1,
                            sharex=True,
                            figsize=(12,24),
                            layout="constrained",
                        )
                        for c in range(num_channels):
                            # Puisque ce sont des signaux 1D, on les trace directement.
                            signal_sample = generated_sample[:, c]
                            signal_perfect = perfect_rir_sample[:, c] 
                            signal_real = real_rir_sample[:, c]
                            axes[c].plot(signal_sample, label="Generated RIR")
                            axes[c].plot(signal_perfect, label="Perfect RIR")
                            axes[c].plot(signal_real, label="Real RIR",linestyle = '-', linewidth=0.5)
                            axes[c].set_ylim(bottom=-1.5, top=1.5)
                        
                        axes[-1].set_xlabel("Time")
                        axes[-1].legend()
                        fig.suptitle("Signal per channels")
                        # fig.tight_layout()
                        fig.subplots_adjust(hspace=0)
                        fabric.logger.experiment.add_figure(f"sample_at_step_{step}", fig)
                        plt.close()
                        del fig, axes, signal_sample, signal_perfect, signal_real, perfect_rir_sample, generated_sample, real_rir_sample
                        torch.cuda.empty_cache()
                    
                
                
            # Gather loss from all processes, log value only on process with rank 0
            if fabric.is_global_zero :
                # for j in range(int(rir_chunks_count)):
                #     all_losses[j] = np.asarray(all_losses[j])
                #     all_losses[j] = np.mean(all_losses[j])
                #     fabric.log(f"evaluation loss{j}", all_losses[j].item(), step)
                #     logging.info("étape: %d, loss évaluation: %.5e", step, all_losses[j].item())
                all_losses = np.asarray(all_losses)
                all_losses = np.mean(all_losses)
                fabric.log(f"evaluation loss", all_losses.item(), step)
                logging.info("étape: %d, loss évaluation: %.5e", step, all_losses.item())




                

            