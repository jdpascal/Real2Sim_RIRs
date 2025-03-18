"""Training and evaluation"""

import argparse
import logging
import os
import runpy
from pathlib import Path

import ml_collections
from lightning import Fabric

import run_lib_jd as run_lib


def load_config(config_path) -> ml_collections.ConfigDict:
    """
    Charge la configuration depuis un fichier Python.
    Le fichier doit définir une fonction `get_config()` qui retourne un ml_collections.ConfigDict.
    """
    config_module = runpy.run_path(config_path)
    logging.info("config module loaded with runpy")
    if "get_config" in config_module:
        logging.info("get_config")
        return config_module["get_config"]()
    elif "config" in config_module:
        logging.info("config")
        return config_module["config"]
    else:
        raise ValueError(
            f"Le fichier {config_path} ne contient ni 'get_config' ni 'config'."
        )


def parse_args() -> argparse.Namespace:
    """
    Sets up parser and parses input arguments, raises errors if provided arguments are invalid.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Training and evaluation script.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers le fichier de configuration Python.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="Répertoire de travail.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        required=True,
        help="Mode d'exécution : train ou eval.",
    )
    parser.add_argument(
        "--eval-folder",
        dest="eval_folder",
        type=str,
        default="eval",
        help="Nom du dossier pour stocker les résultats d'évaluation.",
    )
    parser.add_argument(
        "--ddp-nodes",
        dest="ddp_nodes",
        type=int,
        default=1,
        help="Number of nodes used for parallelism",
    )
    parser.add_argument(
        "--ddp-devices-per-node",
        dest="ddp_devices_per_node",
        type=int,
        default=1,
        help="Number of devices on one node",
    )
    return parser.parse_args()


def main():
    """
    Script entrypoint
    """
    # Parse input arguments and setup logger
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s: [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s) - %(message)s",
        level=logging.DEBUG,
    )
    # Création du répertoire de travail
    os.makedirs(args.workdir, exist_ok=True)

    # Charger la configuration depuis le fichier Python
    print("loading config")
    config = load_config(args.config)

    # Configure Fabric to take care of handling precision, parallelisation, etc
    # TODO: Make precision an argument or a config parameter
    fabric = Fabric(
        precision="bf16-true",
        num_nodes=args.ddp_nodes,
        devices=args.ddp_devices_per_node,
    )
    fabric.launch()

    # Exécuter le pipeline en fonction du mode choisi
    if args.mode == "train":
        run_lib.train(config, Path(args.workdir), fabric)
    elif args.mode == "eval":
        run_lib.evaluate(config, args.workdir, args.eval_folder)
    else:
        raise ValueError(f"Mode {args.mode} non reconnu.")


if __name__ == "__main__":
    main()
