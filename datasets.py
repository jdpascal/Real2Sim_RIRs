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

import logging
import os
import numpy as np
from PIL import Image
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pathlib import Path
import gzip

# -------------------------
# Fonctions de normalisation
# -------------------------
def get_data_scaler(config):
    """Normalise les données dans [0, 1] vers [-1, 1] si demandé."""
    if config.data.centered:
        return lambda x: x * 2 - 1
    else:
        return lambda x: x

def get_data_inverse_scaler(config):
    """Inverse de la normalisation."""
    if config.data.centered:
        return lambda x: (x + 1) / 2
    else:
        return lambda x: x



def no_geometric_attenuation(rir, len):
    t = np.linspace(0, len, len) + 0.05
    rir["real_rir"] = rir["real_rir"] * np.sqrt(np.sqrt(t))
    # rir["real_rir"] = rir["real_rir"] / 2 / np.max(rir["real_rir"]) + 0.5
    rir["perfect_rir"] = rir["perfect_rir"] * np.sqrt(np.sqrt(t))
    # rir["perfect_rir"] = rir["perfect_rir"] / 2 /  np.max(rir["perfect_rir"]) + 0.5
    return( rir )

def choose_random_channel(rir):
    """
    Select a random channel in the RIR.
    """
    random_channel = np.random.randint(0, 32)
    rir0 = rir["real_rir"][random_channel, None, :]
    rir1 = rir["perfect_rir"][random_channel, None, :]
    for k in range(32):
        if k == random_channel :
            continue
        rir["real_rir"][k] = rir0
        rir["perfect_rir"][k] = rir1

    return rir

# -------------------------
# Dataset personnalisé pour MultiRIR
# -------------------------
class MultiRIRDataset(Dataset):
    
    def __init__(self, root_dir: str, config, mode="train", transform=None):
        self.root_dir = root_dir
        self.config = config
        self.mode = mode
        self.transform = transform

        # total_samples_count = config.data.total_rir_samples_count
        # self.rir_chunks_count = total_samples_count // config.data.rir_samples_count
        self.beginning = config.data.begining
        
        total = config.data.num_room * config.data.pos_per_room
        train_max_index = int(total * 0.90)
        ## Except for evaluation dataset, we want to run over all the rooms, so we set train_max_index to 1
        if root_dir == "./dataset_ircam/" or root_dir == "./dataset_genelec_8030_near_measure_eval/" or root_dir == "./dataset_genelec_8020/" or root_dir == "./dataset_measurement_cerema/":
            logging.info("changing size of evaluation dataset")
            train_max_index = 1
        
        if mode == "train":
            self.indices = np.arange(stop=train_max_index)
        else:
            self.indices = np.arange(start=train_max_index , stop=total)

        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        room_index = self.indices[idx]
        # if self.mode == "train":
            # room_index = self.indices[idx]
            # chunk_index = 0
        # else:
        #     room_index = self.indices[idx] // self.rir_chunks_count
        #     chunk_index = self.indices[idx] % self.rir_chunks_count
        
        # Get room number and position number from room index using euclidean division
        room_number = int(room_index / self.config.data.pos_per_room)
        pos_number = room_index % self.config.data.pos_per_room
        room_filename = f"room_{room_number}_{pos_number}.json.gz"

        with gzip.open(Path(self.root_dir) / room_filename, 'rb') as json_file:
            room_data = json.loads(json_file.read().decode('utf-8'))
            if self.mode == "train":
                sample = {
                    'perfect_rir': np.array(room_data['perfect_rir'])[
                        :,self.beginning : self.beginning + self.config.data.rir_samples_count
                    ],
                    'real_rir': np.array(room_data['real_rir'])[
                        :,self.beginning  : self.beginning + self.config.data.rir_samples_count
                    ],}
            else:
                sample = {
                    'perfect_rir': np.array(room_data['perfect_rir'])[
                        :,self.beginning: self.beginning + self.config.data.rir_samples_count
                    ],
                    'real_rir': np.array(room_data['measurement_rir'])[
                        :,self.beginning : self.beginning + self.config.data.rir_samples_count
                    ],
                    # 'rir_chunk_index': chunk_index,
                    'geometry': room_data['geometry'],
                    'verite' : np.array(room_data['verite']),
                    'ordre' : np.array(room_data['ordre'])
                }
                sample['real_rir'] = sample['real_rir'] / np.max(sample['real_rir'])
                sample['perfect_rir'] = sample['perfect_rir'] / np.max(sample['perfect_rir'])

        if self.transform:
            sample = self.transform(sample)
            sample['perfect_rir'] = sample['perfect_rir'][:, None, :].astype(np.float32)
            sample['real_rir'] = sample['real_rir'][:, None, :].astype(np.float32)
        else:
            # Reshape data, although this could (should?) be done using a Transform
            # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
            sample['perfect_rir'] = sample['perfect_rir'][:, None, :].astype(np.float32)
            sample['real_rir'] = sample['real_rir'][:, None, :].astype(np.float32)

        return sample

# -------------------------
# Fonction principale pour créer les DataLoaders
# -------------------------
def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """
    Crée des DataLoaders pour l'entraînement et l'évaluation.

    Args:
        config: Objet de configuration contenant les paramètres (ex. config.data.dataset, config.data.image_size, etc.)
        uniform_dequantization: Si True, ajoute du bruit uniforme aux images.
        evaluation: Si True, utilise les paramètres d'évaluation (ex. désactivation des augmentations).

    Returns:
        train_loader, eval_loader, dataset_builder (None ici)
    """
    # Calcul du batch_size
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    len = config.data.rir_samples_count
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if batch_size % n_devices != 0:
        raise ValueError(f"Le batch size ({batch_size}) doit être divisible par le nombre de devices ({n_devices}).")

    # En fonction du dataset, on définit les transformations et on crée le dataset
    if config.data.dataset == 'Multichannel_RIR':
        # On charge le fichier NPZ et on crée un dataset personnalisé.
        train_dataset = MultiRIRDataset(
            root_dir=config.data.npz_path,
            config=config,
            mode="train",
            transform=transforms.Lambda(lambda x: no_geometric_attenuation(x,len))
        )
        eval_dataset = MultiRIRDataset(
            root_dir=config.data.npz_path,
            config=config,
            mode="eval",
            transform=transforms.Lambda(lambda x: no_geometric_attenuation(x,len))
        )
    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} non supporté.")
    train_loader = 0
    # Création des DataLoaders
    if config.data.npz_path != "./dataset_ircam/" and config.data.npz_path!= "./dataset_genelec_8030_near_measure_eval/" and config.data.npz_path!= "./dataset_genelec_8020/" and config.data.npz_path != "./dataset_measurement_cerema/":  ### your dataset test
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    return train_loader, eval_loader, None
