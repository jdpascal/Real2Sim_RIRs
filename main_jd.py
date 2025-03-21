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

"""Training and evaluation"""
"""Training and evaluation"""

import run_lib
import argparse
import json
import logging
import os
import sys


def load_config(config_path):
    # Ici, on suppose que le fichier de config est au format JSON.
    # Vous pouvez adapter cette fonction pour utiliser un autre format si nécessaire.
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main():
    logger = logging.getLogger()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    logger.info("Program start")
    
    parser = argparse.ArgumentParser(description="Training and evaluation script.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration (JSON file).")
    parser.add_argument("--workdir", type=str, required=True, help="Working directory.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True, help="Running mode: train or eval.")
    parser.add_argument("--eval_folder", type=str, default="eval", help="Folder name for storing evaluation results.")
    args = parser.parse_args()
    
    logger.info("Loading config...")
    logger.info("Config path %s", args.config)

    # Chargement de la configuration depuis le fichier JSON
    config = load_config(args.config)

    # Création du répertoire de travail
    os.makedirs(args.workdir, exist_ok=True)

    # Lancement du pipeline selon le mode choisi
    if args.mode == "train":
        run_lib.train(config, args.workdir)
    elif args.mode == "eval":
        run_lib.evaluate(config, args.workdir, args.eval_folder)
    else:
        raise ValueError(f"Mode {args.mode} not recognized.")

if __name__ == "__main__":
    main()
