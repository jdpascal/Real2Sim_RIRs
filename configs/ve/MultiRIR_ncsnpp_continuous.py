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

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

import ml_collections
import torch


def get_config() -> ml_collections.ConfigDict:
    """
    Configuration for model shape, hyper parameters, training and eval options, etc

    Returns:
        ml_collections.ConfigDict: Configuration object
    """
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 2
    training.n_iters = 60000
    training.snapshot_freq = 2000
    training.log_freq = 100
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 5000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.sde = "ouvesde"

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 2
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.33
    sampling.method = "pc"
    sampling.predictor = "none"
    sampling.corrector = "ald"

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 25
    # for now only support batch size of 1
    evaluate.batch_size = 1
    evaluate.enable_sampling = True
    evaluate.num_samples = 500
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"

    # data
    config.data = data = ml_collections.ConfigDict()
    data.random_flip = True
    data.uniform_dequantization = False
    data.centered = False
    data.dataset = "MultiRIR"
    data.rir_samples_count = 256
    data.total_rir_samples_count = 1024 # should be a multiple of rir_samples_count
    data.begining = 0
    data.first_stride_convolution = 1 # best results with 1, because it keeps the best resolution
    data.image_size = data.rir_samples_count / data.first_stride_convolution 
    data.channels = 32
    data.tfrecords_path = "./dat"
    data.num_channels = 1
    data.npz_path = "./dataset_source_back/"
    data.num_room = 15000
    data.pos_per_room = 10

    # model
    config.model = model = ml_collections.ConfigDict()
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.name = "ncsnpp"
    model.sigma_max = 1.0
    model.sigma_min = 0.1
    model.num_scales = 200
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "elu"  # "lrelu"
    model.nf = int(data.rir_samples_count / data.first_stride_convolution / 2)
    model.ch_mult = (2,2,4,4,4,4) #, 2, 2, 2
    model.num_res_blocks = 3
    # model.attn_resolutions = (16,)
    model.attn_resolutions = (32,8)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "output_skip"
    model.progressive_input = "input_skip"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.fourier_scale = 2
    model.conv_size = 3



    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 42
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
    elif torch.mps.is_available():
        config.device = torch.device("mps")
    else:
        config.device = torch.device("cpu")


    return config
