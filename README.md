# BRIDGING THE MEASUREMENT–SIMULATION GAP IN ROOM ACOUSTICS WITH REAL2SIM DIFFUSION

** this code is very inspired by [Yang Song's Score SDE Pytorch repository](https://github.com/yang-song/score_sde_pytorch)**

--------------------

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/score-based-generative-modeling-through-1/image-generation-on-cifar-10)](https://paperswithcode.com/sota/image-generation-on-cifar-10?p=score-based-generative-modeling-through-1)

This repo contains a PyTorch implementation for the paper 

by Jean-Daniel PASCAL PRIETO, Antoine DELEFORGE, Cédric FOY, Marceau Tonelli

--------------------

We propose a unified framework that generalizes and improves previous work on score-based generative models through the lens of stochastic differential equations (SDEs). In particular, we can transform data to a simple noise distribution with a continuous-time stochastic process described by an SDE. This SDE can be reversed for sample generation if we know the score of the marginal distributions at each intermediate time step, which can be estimated with score matching. The basic idea is captured in the figure below:

![schematic](assets/schematic.jpg)


## What does this code do?
Aside from the **NCSN++** model in our paper, this codebase also re-implements many previous score-based models in one place, including **NCSN** from [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600), **NCSNv2** from [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011), and **DDPM** from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). 

It supports training new models, evaluating the sample quality and likelihoods of existing models. We carefully designed the code to be modular and easily extensible to new SDEs, predictors, or correctors.


## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval
  --workdir: Working directory
```

* `config` is the path to the config file. Our prescribed config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the path of a config file is a combination of the following dimensions:
  *  dataset: One of "./dataset_genelec_8030_near_measure_eval/", "./dataset_ircam/".
  * model: `ncsnpp`
  * continuous: train the model with continuously sampled time steps. 

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` . When set to "eval", it can do an arbitrary combination of the following




## References

If you find the code useful for your research, please consider citing
```bib

```

