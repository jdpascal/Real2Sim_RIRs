import ml_collections
import torch


def get_default_configs() -> ml_collections.ConfigDict:
    """
    Configuration for model shape, hyper parameters, training and eval options, etc

    Returns:
        ml_collections.ConfigDict: Configuration object
    """
    config = ml_collections.ConfigDict()
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 2
    training.n_iters = 300000
    training.snapshot_freq = 10000
    training.log_freq = 1000
    training.eval_freq = 1000
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 2000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.loss_type = "data_prediction" # score_matching denoiser data_prediction
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.33
    sampling.method = "pc"  # pc or ode
    sampling.predictor = "none"
    sampling.corrector = "ald"

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 30
    evaluate.end_ckpt = 30
    # for now only support batch size of 1
    evaluate.batch_size = 1
    evaluate.enable_sampling = True
    evaluate.num_samples = 200
    # evaluate.num_samples = 500
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"
    # set distance_peaks to True to do the sampling on all the samples in the dataset test and calculate error of estimation peaks
    evaluate.distance_peaks = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.random_flip = True
    data.uniform_dequantization = False
    data.centered = False
    data.dataset = "Multichannel_RIR"
    data.rir_samples_count = 1024
    data.total_rir_samples_count = 1024 # should be a multiple of rir_samples_count
    data.begining = 0
    data.image_size = data.rir_samples_count / 4
    data.channels = 32
    data.num_channels = 1
    data.npz_path = "./dataset_genelec_8030_near_measure/"
    # data.npz_path = "./dataset_ircam/"

    data.num_room = 15900
    data.pos_per_room = 10
    data.sample_rate = 16000
    if data.npz_path == "./dataset_ircam/":
        data.num_room = 1
        data.pos_per_room = 10

    # model
    config.model = model = ml_collections.ConfigDict()
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
    model.num_scales = 200  #2
    model.scale_by_sigma = False
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "elu"  # "lrelu"



    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
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
