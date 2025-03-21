# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
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
import torch
from absl import flags
from lightning import Fabric
from ml_collections import ConfigDict
from torch.utils.tensorboard import SummaryWriter

import datasets
import evaluation_jd as evaluation
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
    # Summarises training logs to visualise with tensorboard
    writer = SummaryWriter(
        log_dir=workdir / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # Initialisation du modèle.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    score_model, optimizer = fabric.setup(score_model, optimizer)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    logging.info("Creating checkpoints dirs...")
    # Création des dossiers pour les checkpoints.
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_meta_dir = workdir / "checkpoints-meta" / "checkpoint.pth"
    checkpoint_meta_dir.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Restore checkpoints...")
    # Reprise de l'entraînement si un checkpoint intermédiaire existe.
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state["step"])

    logging.info("Building dataloaders...")
    # Construction des DataLoaders (en version PyTorch).
    train_loader, eval_loader, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
    )

    train_loader = fabric.setup_dataloaders(train_loader)
    eval_loader = fabric.setup_dataloaders(eval_loader)
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    logging.info("Creating normalization and inverse functions...")
    # Création des fonctions de normalisation et d'inverse.
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Configuration de la SDE.
    sde_name = config.training.sde.lower()
    if sde_name == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif sde_name == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif sde_name == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    elif sde_name == "ouvesde":
        sde = sde_lib.OUVESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} inconnu.")

    # Préparation des fonctions de mise à jour pour l'entraînement et l'évaluation.
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde,
        fabric,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        fabric,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # Construction de la fonction de sampling si nécessaire.
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.batch_size,
            config.data.num_channels,
            config.data.rir_samples_count,
            config.data.channels,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    num_train_steps = config.training.n_iters
    logging.info("Début de la boucle d'entraînement à l'étape %d.", initial_step)

    for step in range(initial_step, num_train_steps + 1):
        logging.info("Loop start %d", step)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Si le dataset contient des clés spécifiques (ex. 'perfect_rir' et 'real_rir'), on les traite.
        if "perfect_rir" in batch and "real_rir" in batch:
            perfect_rir = batch["perfect_rir"]
            # print("perfect_rir.shape", perfect_rir.shape)
            # Si nécessaire, ajuster l'ordre des dimensions (ex. HWC -> CHW)
            if (
                perfect_rir.ndim == 4
                and perfect_rir.shape[-1] != config.data.num_channels
            ):
                perfect_rir = perfect_rir.permute(0, 2, 3, 1)
            real_rir = batch["real_rir"]
            # print("perfect_rir.shape", perfect_rir.shape)
            # print("real_rir.shape", real_rir.shape)
            if real_rir.ndim == 4 and real_rir.shape[-1] != config.data.num_channels:
                real_rir = real_rir.permute(0, 2, 3, 1)
            # print("real_rir.shape", real_rir.shape)
            current_batch = (perfect_rir, real_rir)
        else:
            # Pour un dataset classique avec la clé 'image'.
            img = batch["image"]
            if img.ndim == 4 and img.shape[-1] != config.data.num_channels:
                img = img.permute(0, 3, 1, 2)
            # Appliquer la normalisation (si nécessaire).
            current_batch = scaler(img)

        # Exécuter une étape d'entraînement
        loss = train_step_fn(state, current_batch)
        # del current_batch

        if step % config.training.log_freq == 0:
            logging.info("étape: %d, loss entraînement: %.5e", step, loss.item())
            writer.add_scalar("training_loss", loss.item(), step)

        # Sauvegarde d'un checkpoint temporaire pour reprise en cas d'interruption.
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Évaluation périodique sur le jeu de validation.
        if step % config.training.eval_freq == 0:
            try:
                eval_batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_loader)
                eval_batch = next(eval_iter)
            # Ici, on suppose que le dataset d'évaluation renvoie la clé 'image'.
            eval_img_perfect = eval_batch["perfect_rir"].to(fabric.device)
            eval_img_real = eval_batch["real_rir"].to(fabric.device)
            # print("avant permute ? eval perfect, real" , eval_img_perfect.shape, eval_img_real.shape)
            if (
                eval_img_perfect.ndim == 4
                and eval_img_perfect.shape[-1] != config.data.num_channels
            ):
                eval_img_perfect = eval_img_perfect.permute(0, 2, 3, 1)
                eval_img_real = eval_img_real.permute(0, 2, 3, 1)
            # print("apres permute?" , eval_img_perfect.shape, eval_img_real.shape )
            # eval_img = scaler(eval_img)
            eval_img_batch = (eval_img_perfect, eval_img_real)
            eval_loss = eval_step_fn(state, eval_img_batch)
            logging.info("étape: %d, loss évaluation: %.5e", step, eval_loss.item())
            scalar_ = eval_loss.item()
            writer.add_scalar("eval_loss", scalar_, step)

        # Sauvegarde d'un checkpoint complet et génération d'échantillons.
        if (step != 0 and step % config.training.snapshot_freq == 0) or (
            step == num_train_steps
        ):
            # Sauvegarde du checkpoint.
            save_step = step // config.training.snapshot_freq
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth")
            save_checkpoint(ckpt_path, state)

            # Génération d'échantillons.
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sampling_fn = sampling.get_sampling_fn(
                    config,
                    sde,
                    sampling_shape,
                    inverse_scaler,
                    sampling_eps,
                    y=real_rir,
                )
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())

                # Sélection du premier exemple du batch pour la comparaison
                perfect_rir_sample = (
                    perfect_rir[0].detach().cpu().numpy()
                )  # forme: (epaisseur=1, longueur, canaux)
                generated_sample = (
                    sample[0].detach().cpu().numpy()
                )  # forme: (epaisseur=1, longueur, canaux)
                # Suppression de la dimension "épaisseur" (qui vaut 1)
                perfect_rir_sample = np.squeeze(
                    perfect_rir_sample, axis=0
                )  # devient (longueur, canaux)
                generated_sample = np.squeeze(
                    generated_sample, axis=0
                )  # devient (longueur, canaux)

                # On trace jusqu'à 32 canaux (ou le nombre maximum de canaux disponibles)
                plt.ioff()
                num_channels = min(32, perfect_rir_sample.shape[1])
                fig, axes = plt.subplots(
                    nrows=num_channels, ncols=1, sharex=True, figsize=(6, 12)
                )
                for c in range(num_channels):
                    # Puisque ce sont des signaux 1D, on les trace directement.
                    signal_sample = generated_sample[:, c] / np.max(
                        generated_sample[:, c]
                    )
                    signal_perfect = perfect_rir_sample[:, c]
                    axes[c].plot(signal_sample, label="Channel sample")
                    axes[c].plot(signal_perfect, label="Channel perfect")
                    axes[c].set_ylim(bottom=-1.5, top=1.5)

                axes[-1].set_xlabel("Time")
                axes[-1].legend()
                fig.suptitle("Signal per channels")
                fig.tight_layout()
                fig.subplots_adjust(hspace=0)
                writer.add_figure(f"sample_at_step_{step}", fig)
                plt.close()

    writer.close()


def evaluate(config, workdir, eval_folder="eval"):
    """
    Évalue les modèles entraînés.

    Args:
        config: Objet de configuration.
        workdir: Répertoire de travail contenant les checkpoints.
        eval_folder: Sous-dossier pour stocker les résultats d'évaluation.
    """
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Chargement des DataLoaders en mode évaluation.
    _, eval_loader, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        evaluation=True,
    )

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialisation du modèle, de l'optimizer et de l'EMA.
    score_model = mutils.create_model(config)
    score_model = ncsnpp.NCSNpp(config.model)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Configuration de la SDE.
    sde_name = config.training.sde.lower()
    if sde_name == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
    if sde_name == "ouvesde":
        sde = sde_lib.OUVESDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif sde_name == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif sde_name == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} inconnu.")

    # Création de la fonction d'évaluation de la loss si activée.
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting
        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(
            sde,
            train=False,
            optimize_fn=optimize_fn,
            reduce_mean=reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
        )

    # Pour le calcul de la vraisemblance (bits/dim), on choisit le dataset.
    train_loader_bpd, eval_loader_bpd, _ = datasets.get_dataset(
        config, uniform_dequantization=True, evaluation=True
    )
    if config.eval.bpd_dataset.lower() == "train":
        ds_bpd = train_loader_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == "test":
        ds_bpd = eval_loader_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"Dataset bpd {config.eval.bpd_dataset} non reconnu.")

    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

    if config.eval.enable_sampling:
        sampling_shape = (
            config.eval.batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    inceptionv3 = config.data.image_size >= 256
    # On suppose que evaluation.get_inception_model renvoie un modèle Inception adapté en PyTorch.
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("Début de l'évaluation à partir du checkpoint %d.", begin_ckpt)

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
            state = restore_checkpoint(ckpt_filename, state)
        except Exception:
            logging.warning(
                "Problème lors du chargement du checkpoint, nouvelle tentative dans 60s..."
            )
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_filename, state)
            except Exception:
                time.sleep(120)
                state = restore_checkpoint(ckpt_filename, state)

        ema.copy_to(score_model.parameters())

        # Évaluation de la loss sur l'ensemble du dataset d'évaluation.
        if config.eval.enable_loss:
            all_losses = []
            for i, batch in enumerate(eval_loader):
                img = batch["image"]
                if img.ndim == 4 and img.shape[-1] != config.data.num_channels:
                    img = img.permute(0, 3, 1, 2)
                img = scaler(img)
                loss_val = eval_step(state, img)
                all_losses.append(loss_val.item())
                if (i + 1) % 1000 == 0:
                    logging.info(
                        "Évaluation loss, étape %d sur %d", i + 1, len(eval_loader)
                    )
            all_losses = np.asarray(all_losses)
            loss_filepath = os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz")
            with open(loss_filepath, "wb") as f:
                np.savez_compressed(
                    f, all_losses=all_losses, mean_loss=all_losses.mean()
                )

        # Calcul de la vraisemblance (bits/dim) si activé.
        if config.eval.enable_bpd:
            bpds = []
            for repeat in range(bpd_num_repeats):
                bpd_iter = iter(ds_bpd)
                for batch_id, batch in enumerate(bpd_iter):
                    img = batch["image"]
                    if img.ndim == 4 and img.shape[-1] != config.data.num_channels:
                        img = img.permute(0, 3, 1, 2)
                    img = scaler(img)
                    bpd_val = likelihood_fn(score_model, img)[0]
                    bpd_val = bpd_val.detach().cpu().numpy().reshape(-1)
                    bpds.extend(bpd_val)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, moyenne bpd: %.6f",
                        ckpt,
                        repeat,
                        batch_id,
                        np.mean(np.asarray(bpds)),
                    )
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    bpd_filepath = os.path.join(
                        eval_dir,
                        f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz",
                    )
                    with open(bpd_filepath, "wb") as f:
                        np.savez_compressed(f, bpds=bpds)

        # Génération d'échantillons et calcul des métriques IS/FID/KID si activés.
        if config.eval.enable_sampling:
            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            for r in range(num_sampling_rounds):
                logging.info("Sampling -- ckpt: %d, round: %d", ckpt, r)
                this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
                os.makedirs(this_sample_dir, exist_ok=True)
                samples, n = sampling_fn(score_model)
                samples_np = np.clip(
                    samples.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
                ).astype(np.uint8)
                samples_np = samples_np.reshape(
                    (
                        -1,
                        config.data.image_size,
                        config.data.image_size,
                        config.data.num_channels,
                    )
                )
                samples_filepath = os.path.join(this_sample_dir, f"samples_{r}.npz")
                np.savez_compressed(samples_filepath, samples=samples_np)

                gc.collect()
                # On suppose que evaluation.run_inception_distributed est adapté à PyTorch.
                latents = evaluation.run_inception_distributed(
                    samples_np, inception_model, inceptionv3=inceptionv3
                )
                gc.collect()
                stats_filepath = os.path.join(this_sample_dir, f"statistics_{r}.npz")
                with open(stats_filepath, "wb") as f:
                    np.savez_compressed(
                        f, pool_3=latents["pool_3"], logits=latents["logits"]
                    )

            # Récupération de toutes les statistiques pour le calcul des métriques.
            all_logits = []
            all_pools = []
            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
            stats_files = glob.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
            for stat_file in stats_files:
                with open(stat_file, "rb") as f:
                    stat = np.load(f)
                    if not inceptionv3:
                        all_logits.append(stat["logits"])
                    all_pools.append(stat["pool_3"])

            if not inceptionv3:
                all_logits = np.concatenate(all_logits, axis=0)[
                    : config.eval.num_samples
                ]
            all_pools = np.concatenate(all_pools, axis=0)[: config.eval.num_samples]

            # Chargement des statistiques du dataset de référence.
            data_stats = evaluation.load_dataset_stats(config)
            data_pools = data_stats["pool_3"]

            # Calcul des métriques.
            # Vous devrez remplacer ces appels par des fonctions PyTorch adaptées (ex. via pytorch-fid).
            if not inceptionv3:
                inception_score = evaluation.compute_inception_score(all_logits)
            else:
                inception_score = -1

            fid = evaluation.compute_fid(data_pools, all_pools)
            kid = evaluation.compute_kid(data_pools, all_pools)

            logging.info(
                "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e",
                ckpt,
                inception_score,
                fid,
                kid,
            )

            report_filepath = os.path.join(eval_dir, f"report_{ckpt}.npz")
            with open(report_filepath, "wb") as f:
                np.savez_compressed(f, IS=inception_score, fid=fid, kid=kid)
