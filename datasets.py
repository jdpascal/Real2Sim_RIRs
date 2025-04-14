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
        
        total = config.data.num_room * config.data.pos_per_room
        train_max_index = int(total * 0.8)
        
        if mode == "train":
            self.indices = np.arange(stop=train_max_index)
        else:
            self.indices = np.arange(start=train_max_index, stop=total)
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        room_index = self.indices[idx]
        
        # Get room number and position number from room index using euclidean division
        room_number = int(room_index / self.config.data.pos_per_room)
        pos_number = room_index % self.config.data.pos_per_room
        room_filename = f"room_{room_number}_{pos_number}.json"

        with open(Path(self.root_dir) / room_filename, 'r') as json_file:
            room_data = json.load(json_file)
            sample = {
                'perfect_rir': np.array(room_data['perfect_rir'])[:,:self.config.data.rir_samples_count],
                'real_rir': np.array(room_data['real_rir'])[:,:self.config.data.rir_samples_count],
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

# class MultiRIRDataset(Dataset):
#     def __init__(self, npz_path, split='train', config=None, transform=None):
#         if not os.path.exists(npz_path):
#             raise ValueError(f"Le fichier NPZ {npz_path} n'existe pas.")
        
#         # data = {'real_rir': [], 'perfect_rir': []}
#         # for room_index in range(config.data.num_room):
#         #     for position_index in range(config.data.pos_per_room):
#         #         with open(npz_path  + f"room_{room_index}_{position_index}.json", 'r') as json_file:
#         #             data_room = json.load(json_file)
#         #             data['real_rir'].append(data_room['real_rir'])
#         #             data['perfect_rir'].append(data_room['perfect_rir'])
#         #             # data['condition'].append(data_room['condition'])  if 'condition' in data_room.keys() else None

#         with open(npz_path + "data.json", "r") as json_file:
#             data = json.load(json_file)
        
#         self.X = data['perfect_rir']
#         self.Y = data['real_rir']
#         self.condition = data['condition'] if 'condition' in data.keys() else None

#         total = len(self.X)
#         split_idx = int(0.8 * total)  # 80% pour l'entraînement, 20% pour l'évaluation
#         if split == 'train':
#             self.indices = np.arange(split_idx)
#         else:
#             self.indices = np.arange(split_idx, total)

#         self.transform = transform

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         i = self.indices[idx]
#         sample = {
#             'perfect_rir': self.X[i],
#             'real_rir': self.Y[i]
#         }
#         if self.condition is not None:
#             sample['condition'] = self.condition[i]
#         if self.transform:
#             sample = self.transform(sample)
#         else:
#             # Conversion basique en tenseurs
#             sample['perfect_rir'] = torch.FloatTensor(sample['perfect_rir'])
#             sample['real_rir'] = torch.FloatTensor(sample['real_rir'])
#             sample['perfect_rir'] = sample['perfect_rir'][:, None, :]
#             sample['real_rir'] = sample['real_rir'][:, None, :]
#             if 'condition' in sample:
#                 sample['condition'] = torch.FloatTensor(sample['condition'])
#         return sample

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

    # Fonction utilitaire pour éventuellement ajouter la déquantification uniforme
    def maybe_dequantize():
        if uniform_dequantization:
            return transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256)
        else:
            return transforms.Lambda(lambda x: x)

    # En fonction du dataset, on définit les transformations et on crée le dataset
    if config.data.dataset == 'CIFAR10':
        transform_list = [
            transforms.Resize((config.data.image_size, config.data.image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC)
        ]
        if config.data.random_flip and not evaluation:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend([
            transforms.ToTensor(),
            maybe_dequantize()
        ])
        transform = transforms.Compose(transform_list)
        train_dataset = datasets.CIFAR10(root=config.data.data_dir,
                                         train=True, transform=transform, download=True)
        eval_dataset = datasets.CIFAR10(root=config.data.data_dir,
                                        train=False, transform=transform, download=True)

    elif config.data.dataset == 'SVHN':
        transform_list = [
            transforms.Resize((config.data.image_size, config.data.image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC)
        ]
        if config.data.random_flip and not evaluation:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend([
            transforms.ToTensor(),
            maybe_dequantize()
        ])
        transform = transforms.Compose(transform_list)
        train_dataset = datasets.SVHN(root=config.data.data_dir,
                                      split='train', transform=transform, download=True)
        eval_dataset = datasets.SVHN(root=config.data.data_dir,
                                     split='test', transform=transform, download=True)

    elif config.data.dataset == 'CELEBA':
        transform_list = [
            transforms.CenterCrop(140),
            transforms.Resize((config.data.image_size, config.data.image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC)
        ]
        if config.data.random_flip and not evaluation:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend([
            transforms.ToTensor(),
            maybe_dequantize()
        ])
        transform = transforms.Compose(transform_list)
        train_dataset = datasets.CelebA(root=config.data.data_dir,
                                        split='train', transform=transform, download=True)
        eval_dataset = datasets.CelebA(root=config.data.data_dir,
                                       split='valid', transform=transform, download=True)

    elif config.data.dataset == 'LSUN':
        # Pour LSUN, le paramètre config.data.category doit être défini (ex. 'bedroom')
        if config.data.image_size == 128:
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                maybe_dequantize()
            ])
        else:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: crop_resize(img, config.data.image_size)),
                transforms.ToTensor(),
                maybe_dequantize()
            ])
        train_dataset = datasets.LSUN(root=config.data.data_dir,
                                      classes=[config.data.category], transform=transform, split='train')
        eval_dataset = datasets.LSUN(root=config.data.data_dir,
                                     classes=[config.data.category], transform=transform, split='val')

    elif config.data.dataset == 'MultiRIR':
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

    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        raise NotImplementedError(f"Le dataset {config.data.dataset} n'est pas encore implémenté pour PyTorch.")

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
