import logging
import os
import numpy as np
from PIL import Image
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pathlib import Path

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

# -------------------------
# Fonctions de recadrage/redimensionnement
# -------------------------
def crop_resize(image, resolution):
    """
    Recadre au centre une image PIL pour obtenir un carré de taille minimale,
    puis la redimensionne à (resolution, resolution) avec interpolation bicubique.
    """
    w, h = image.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))
    return image.resize((resolution, resolution), resample=Image.BICUBIC)

def no_geometric_attenuation(rir, len):
    t = np.linspace(0, len, len)
    rir["real_rir"] = rir["real_rir"] #* np.sqrt(t)
    rir["real_rir"] = rir["real_rir"] / 2 / np.max(rir["real_rir"]) + 0.5
    rir["perfect_rir"] = rir["perfect_rir"] #* np.sqrt(t)
    rir["perfect_rir"] = rir["perfect_rir"] / 2 /  np.max(rir["perfect_rir"]) + 0.5
    return( rir )

# -------------------------
# Dataset personnalisé pour MultiRIR
# -------------------------
class MultiRIRDataset(Dataset):
    
    def __init__(self, root_dir: str, config, mode="train", transform=None):
        self.root_dir = root_dir
        self.config = config
        self.mode = mode
        self.transform = transform

        total_samples_count = config.data.total_rir_samples_count
        self.rir_chunks_count = total_samples_count // config.data.rir_samples_count
        self.beginning = config.data.begining
        
        total = config.data.num_room * config.data.pos_per_room
        train_max_index = int(total * 0.8)
        
        if mode == "train":
            self.indices = np.arange(stop=train_max_index)
        else:
            self.indices = np.arange(start=train_max_index, stop=total) * self.rir_chunks_count
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        if self.mode == "train":
            room_index = self.indices[idx]
            chunk_index = 0
        else:
            room_index = self.indices[idx // self.rir_chunks_count]
            chunk_index = room_index % self.rir_chunks_count
        
        # Get room number and position number from room index using euclidean division
        room_number = int(room_index / self.config.data.pos_per_room)
        pos_number = room_index % self.config.data.pos_per_room
        room_filename = f"room_{room_number}_{pos_number}.json"

        with open(Path(self.root_dir) / room_filename, 'r') as json_file:
            room_data = json.load(json_file)
            sample = {
                'perfect_rir': np.array(room_data['perfect_rir'])[
                    :,self.beginning + chunk_index * self.config.data.rir_samples_count : self.beginning + (chunk_index + 1) * self.config.data.rir_samples_count
                ],
                'real_rir': np.array(room_data['real_rir'])[
                    :,self.beginning + chunk_index * self.config.data.rir_samples_count : self.beginning + (chunk_index + 1) * self.config.data.rir_samples_count
                ],
                'rir_chunk_index': chunk_index,
            }

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
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if batch_size % n_devices != 0:
        raise ValueError(f"Le batch size ({batch_size}) doit être divisible par le nombre de devices ({n_devices}).")

    # En fonction du dataset, on définit les transformations et on crée le dataset
    if config.data.dataset == 'MultiRIR':
        # On charge le fichier NPZ et on crée un dataset personnalisé.
        train_dataset = MultiRIRDataset(
            root_dir=config.data.npz_path,
            config=config,
            mode="train",
            # transform=transforms.Lambda(lambda x: no_geometric_attenuation(x, config.data.rir_samples_count))
        )
        eval_dataset = MultiRIRDataset(
            root_dir=config.data.npz_path,
            config=config,
            mode="eval",
            # transform=transforms.Lambda(lambda x: no_geometric_attenuation(x, config.data.rir_samples_count))
        )
    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} non supporté.")

    # Création des DataLoaders
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
