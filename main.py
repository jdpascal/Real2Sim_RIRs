"""Training and evaluation"""

import sys
import run_lib_jd as run_lib
import argparse
import runpy
import logging
import os

def load_config(config_path):
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
        raise ValueError(f"Le fichier {config_path} ne contient ni 'get_config' ni 'config'.")

def main():
    parser = argparse.ArgumentParser(description="Training and evaluation script.")
    parser.add_argument("--config", type=str, required=True,
                        help="Chemin vers le fichier de configuration Python.")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Répertoire de travail.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True,
                        help="Mode d'exécution : train ou eval.")
    parser.add_argument("--eval_folder", type=str, default="eval",
                        help="Nom du dossier pour stocker les résultats d'évaluation.")
    args = parser.parse_args()

    # Charger la configuration depuis le fichier Python
    print("loading config")
    config = load_config(args.config)

    # Création du répertoire de travail
    os.makedirs(args.workdir, exist_ok=True)

    # Configuration du logger pour écrire à la fois sur la console et dans un fichier
    log_file = os.path.join(args.workdir, 'stdout.txt')
    gfile_stream = open(log_file, 'w')
    handler = logging.StreamHandler(gfile_stream)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s) - %(message)s')
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # Exécuter le pipeline en fonction du mode choisi
    if args.mode == "train":
        run_lib.train(config, args.workdir)
    elif args.mode == "eval":
        run_lib.evaluate(config, args.workdir, args.eval_folder)
    else:
        raise ValueError(f"Mode {args.mode} non reconnu.")


if __name__ == "__main__":
    main()
