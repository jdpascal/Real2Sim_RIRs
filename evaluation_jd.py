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

"""Utility functions for computing FID/Inception scores."""
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

def get_inception_model(inceptionv3=True):
    """
    Retourne un modèle Inception adapté pour l'extraction des features (pool3).
    Actuellement, seule la version Inception V3 est supportée.
    """
    if inceptionv3:
        model = models.inception_v3(pretrained=True, transform_input=False)
        model.aux_logits = False
        # Remplacer la couche finale par une identité pour obtenir directement
        # les features issues de l'adaptive average pooling (dimension 2048)
        model.fc = nn.Identity()
        model.eval()
        return model
    else:
        raise NotImplementedError("Seule la version inceptionv3 est supportée en PyTorch.")

def load_dataset_stats(config):
    """
    Charge les statistiques pré-calculées pour le calcul du FID/Inception Score.
    """
    if config.data.dataset == 'CIFAR10':
        filename = 'assets/stats/cifar10_stats.npz'
    elif config.data.dataset == 'CELEBA':
        filename = 'assets/stats/celeba_stats.npz'
    elif config.data.dataset == 'LSUN':
        filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
    else:
        raise ValueError(f'Statistiques du dataset {config.data.dataset} introuvables.')
    stats = np.load(filename)
    return stats

def classifier_fn(images, inception_model):
    """
    Applique le modèle Inception aux images et retourne les features aplaties.
    
    Args:
        images: torch.Tensor de forme (N, C, H, W) avec des valeurs dans [0, 255].
        inception_model: modèle PyTorch (ici Inception V3) pour l'extraction des features.
    
    Returns:
        Tenseur des features de forme (N, D) (typiquement D = 2048).
    """
    # Mise à l'échelle de [0, 255] vers [0, 1]
    images = images / 255.0
    # Normalisation avec les statistiques ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = (images - mean) / std
    with torch.no_grad():
        features = inception_model(images)
        features = features.view(features.size(0), -1)
    return features

def run_inception(inputs, inception_model, num_batches=1, inceptionv3=True):
    """
    Exécute le modèle Inception sur les entrées et retourne les features.
    
    Args:
        inputs: torch.Tensor de forme (N, C, H, W) avec des valeurs dans [0, 255].
        inception_model: modèle Inception (voir get_inception_model).
        num_batches: si > 1, divise les entrées en ce nombre de lots pour le traitement.
        inceptionv3: doit être True (seule cette version est supportée ici).
        
    Returns:
        Tenseur contenant les features extraites.
    """
    if num_batches > 1:
        batch_size = inputs.size(0) // num_batches
        features_list = []
        for i in range(num_batches):
            batch = inputs[i * batch_size:(i + 1) * batch_size]
            feat = classifier_fn(batch, inception_model)
            features_list.append(feat)
        features = torch.cat(features_list, dim=0)
    else:
        features = classifier_fn(inputs, inception_model)
    return features

def run_inception_distributed(input_tensor, inception_model, num_batches=1, inceptionv3=True):
    """
    Répartit le calcul du modèle Inception sur les GPU disponibles.
    
    Args:
        input_tensor: torch.Tensor de forme (N, C, H, W) avec des valeurs dans [0, 255].
        inception_model: modèle Inception (obtenu via get_inception_model).
        num_batches: nombre de lots pour le traitement (pour éviter les problèmes de mémoire).
        inceptionv3: doit être True.
    
    Returns:
        Dictionnaire contenant la clé 'pool_3' avec les features extraites.
        La clé 'logits' est mise à None car non utilisée ici.
    """
    if torch.cuda.device_count() > 1:
        # Diviser le tenseur d'entrée en autant de morceaux qu'il y a de GPUs
        splits = torch.chunk(input_tensor, torch.cuda.device_count(), dim=0)
        features_list = []
        for i, split in enumerate(splits):
            device = torch.device(f'cuda:{i}')
            split = split.to(device)
            inception_model.to(device)
            feat = run_inception(split, inception_model, num_batches, inceptionv3)
            features_list.append(feat.cpu())
        features = torch.cat(features_list, dim=0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        inception_model.to(device)
        features = run_inception(input_tensor, inception_model, num_batches, inceptionv3)
    return {'pool_3': features, 'logits': None}
