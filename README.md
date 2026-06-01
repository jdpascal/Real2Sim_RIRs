# Bridging the Measurement–Simulation Gap in Room Acoustics with Real2Sim Diffusion

This repository contains the PyTorch implementation of:

**Bridging the Measurement–Simulation Gap in Room Acoustics with Real2Sim Diffusion**  
Jean-Daniel Pascal Prieto, Antoine Deleforge, Cédric Foy, Marceau Tonelli  
[waiting for arXiv publication]

This work investigates a Real2Sim framework for room acoustics. The goal is to transform realistic multichannel room impulse responses (RIRs), including source/microphone directivity and frequency-dependent wall absorption, into simplified canonical RIRs from which geometric information can be more easily extracted.

The code is heavily inspired by [Yang Song's Score SDE PyTorch repository](https://github.com/yang-song/score_sde_pytorch) and [Julius Richter's Speech Enhancement and Dereverberation with Diffusion-based Generative Models repository](https://github.com/sp-uhh/sgmse).

---

## Overview

Room impulse responses measured in real environments contain many effects that are difficult to model exactly, such as loudspeaker and microphone directivity, frequency-dependent wall absorption, and measurement imperfections. These effects can make downstream geometric inference tasks more challenging.

We propose a diffusion-based Real2Sim approach that maps realistic RIRs to simplified RIRs. The simplified representation is designed to preserve the main geometric structure of the acoustic scene while removing part of the complexity introduced by real-world measurement conditions.

In this repository, we focus on the application of image-source localization from multichannel RIRs. Once simplified RIRs are generated, image sources can be estimated using the method proposed by Sprunck et al. The recovered image-source cloud can then be used to infer room geometry.

---

## Main features

This repository supports:

- training diffusion-based Real2Sim models;
- generating synthetic RIR datasets;
- generating RIRs corresponding to the IRCAM measurement campaign;
- evaluating trained models on realistic simulated RIRs;
- evaluating trained models on measured RIRs;
- estimating image-source locations from simplified RIRs;
- aligning estimated image-source clouds with ground-truth annotations.

The implementation is modular and can be extended to new SDEs, predictors, correctors, or data-prediction models.

In the current version of the paper, we use a Schrödinger bridge process to interpolate between realistic and simplified RIRs. We also experimented with an Ornstein–Uhlenbeck Variance-Exploding process, although these results are not reported in the first version of the paper. (inspired by https://arxiv.org/pdf/2208.05830, https://arxiv.org/pdf/2409.10753)

---

## Installation

Create a virtual environment with Python 3.12.8 and install the required dependencies :

```bash
pip install -r requirements.txt
```

The package versions in `requirements.txt` are expected to be compatible with Python 3.12.8.

---

## Modified pyroomacoustics version

This project relies on a modified version of `pyroomacoustics`.

We adapted `pyroomacoustics` to include the directivity pattern of the Genelec 8030 loudspeaker. This directivity pattern was measured in:

> Gallien, A., Prawda, K., & Schlecht, S. J. (2024).  
> *Matching early reflections of simulated and measured RIRs by applying sound-source directivity filters.*  
> Proceedings of the AES 2024 International Acoustics & Sound Reinforcement Conference, Le Mans, France.

We also added a random sign change to the minimum-phase wall filters when generating realistic RIRs. This is motivated by the fact that the angular dependency of wall reflections can influence the phase of the reflection filters. Please use the files RIR_generation/room.py and RIR_generation/ism.py in the pyroomacoustics package.

---

## Dataset generation

### Synthetic dataset

To generate the synthetic RIR dataset, run:

```bash
python RIR_generation/generate_rirs.py
```

Before running the script, check the hyperparameters inside the file and adapt them to the desired dataset configuration.

### IRCAM measurement-based dataset

To generate the dataset corresponding to the IRCAM measurement campaign, run:

```bash
python RIR_generation/generate_rirs_measurement_IRCAM.py
```

The room geometry annotations and measurement positions are defined in this file.

Detailed information about the measurement campaign is provided in:

```text
docs/ircam_measurement_campaign.md
```

---

## Training

Training and evaluation are handled through `main.py`.

Example training command:

```bash
python main.py \
  --config configs/<config_name>.py \
  --mode train \
  --workdir ./experiments/<experiment_name>
```

The `--config` argument specifies the path to a configuration file. Example configuration files are provided in the `configs/` folder and follow the `ml_collections` format.

The `--workdir` argument specifies where checkpoints, samples, logs, and evaluation outputs are stored.

If meta-checkpoints are found in:

```text
<workdir>/checkpoints-meta
```

training resumes automatically from the latest available checkpoint.

---

## Evaluation

Example evaluation command:

```bash
python main.py \
  --config configs/<config_name>.py \
  --mode eval \
  --workdir ./experiments/<experiment_name> \
  --eval_folder eval
```

The `--eval_folder` argument specifies the subfolder of `workdir` where evaluation outputs are stored. This folder contains generated RIRs, intermediate files, checkpoints used for evaluation, and quantitative results.

---

## Image-source localization

To evaluate the geometric information contained in the generated simplified RIRs, we use the image-source localization method from:

```text
https://github.com/Sprunckt/acoustic-sfw.git
```
Then copy our file:

```text
image_source_localization.py
```

into the cloned repository and run it on the generated simplified RIRs.

This produces `.npz` files named:

```text
mes_*.npz
```

containing the estimated image-source clouds and the reconstructed RIR.

---

## Matching and metrics

The estimated image-source clouds are not necessarily aligned with the ground truth because the orientation of the microphone array could not be annotated during the measurement campaign.

We provide a procedure to align the estimated image-source cloud with the expected image-source locations in:

```text
image_source_localization/matching_and_metrics.py
```

This script/notebook allows users to:

- visualize the estimated image-source clouds;
- estimate the microphone-array orientation a posteriori;
- rotate the point cloud around the source–microphone-array axis;
- perform partial matching between estimated and expected image sources;
- compute the final localization metrics.

The alignment relies on the fact that the real source position is accurately estimated. We first superimpose the estimated source and the true source. Then, we estimate the rotation of the point cloud around the source–microphone-array axis using an adapted Hungarian algorithm.

Although the radial and angular matching thresholds are set to 1 m and 20°, the final errors are typically well below these thresholds. These relatively large thresholds are only used to ensure that the correct matching is recovered, even in the presence of measurement errors or missing annotations.

---

## IRCAM measurement campaign

We conducted a measurement campaign to evaluate our framework on real data. The measurements were performed in a shoebox-like room at IRCAM using a Genelec 8030B loudspeaker and an Eigenmike32 microphone array.

The full measurement protocol, acquisition setup, room description, calibration procedure, and image-source alignment strategy are described in:

```text
ircam_measurement_campaign.md
```

---

## Repository structure

```text
configs/                         Training and evaluation configuration files
RIR_generation/                  Scripts for synthetic and measurement-based RIR generation
image_source_localization/       Image-source matching and metric computation
assets/                          Figures used in the README
docs/                            Additional documentation
main.py                          Main entry point for training and evaluation
requirements.txt                 Python dependencies
```

---

## Citation

If you find this code useful for your research, please consider citing our work:

```bibtex
@inproceedings{prieto2026bridging,
  title={Bridging the Measurement--Simulation Gap in Room Acoustics with Real2sim Diffusion},
  author={Prieto, Jean-Daniel Pascal and Deleforge, Antoine and Foy, C{\'e}dric and Tonelli, Marceau},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={14922--14926},
  year={2026},
  organization={IEEE}
}
```

---

## License

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
