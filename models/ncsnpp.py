# coding=utf-8
# pylint: skip-file
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
import logging
from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
ResnetBlockBigGAN_multichannel = layerspp.ResnetBlockBigGANpp_multichannel
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
# conv3x1 = layerspp.conv3x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
logging.basicConfig(level=logging.INFO)


@utils.register_model(name="ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        # self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = (
            config.model.progressive_input.lower()
        )
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            assert config.training.continuous, (
                "Fourier features are only used for continuous training."
            )

            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=config.model.fourier_scale
                )
            )
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        elif resblock_type == "multichannel_biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN_multichannel,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        modules.append(conv3x3(
            config.data.channels * 2,
            int(config.data.rir_samples_count / 2),
            # 256,
            stride=1,
            kernel_size=(15,1),
            padding=(7,0)
        ))

        # Downsampling block

        channels = config.data.num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        logging.debug("NUMBER OF MODULES AFTER DOWNSAMPLING: %d", len(modules))
        in_ch = hs_c[-1] 
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        logging.debug("NUMBER OF MODULES AFTER BOTTLENECK: %d", len(modules))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(
                num_res_blocks + 1
            ):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        modules.append(
            nn.ConvTranspose2d(
                # 256,
                int(config.data.rir_samples_count / 2),
                config.data.channels,
                kernel_size=(15, 1),
                stride=1, ###### put stride 1 to have output size equal to input size ######
                padding=(7, 0),
                output_padding=(0, 0),  # (3,0) if stride_first_convolution == 4 else (1, 0) for stride_first_convolution == 2, else (0, 0) for stride_first_convolution == 1
            )
        )

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, y, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        # print(modules)
        if y is not None:
            # print(x.shape)
            x = torch.cat([x, y], dim=3)
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
            temb = modules[m_idx](temb)
            m_idx += 1
            logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # x = torch.reshape(x, (x.shape[0], x.shape[3], x.shape[2], x.shape[1]))
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        logging.debug(f"START SHAPE: {x.shape}")
        x = x.permute(0, 3, 2, 1)
        logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
        x = modules[m_idx](x)
        m_idx += 1
        # x = torch.reshape(x, (x.shape[0], x.shape[3], x.shape[2], x.shape[1]))
        x = x.permute(0, 3, 2, 1)
        logging.debug(f"AFTER FIRST CONV SHAPE: {x.shape}")
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

        # Downsampling block
        
        logging.debug("------ DOWNSAMPLING BLOCK ------")
        
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x
        logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                # print("residual_block input", hs[-1].shape)
                logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
                # logging.debug( sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad) )
                # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # print("residual_block output", h.shape)

                if h.shape[-1] in self.attn_resolutions:
                    # print("attn_block input", h.shape)
                    logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
                    h = modules[m_idx](h)
                    m_idx += 1
                    # print("attn_block output", h.shape)

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
                    logging.debug(f"bug SHAPE: {h.shape} and hs {hs[-1].shape} and temb {temb.shape}")
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
                    logging.debug(f"shape pyramide {input_pyramid.shape} and {h.shape}")
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        logging.debug("------ BOTTLENECK ------")

        # print(hs[-1].shape)
        # print("m_idx", m_idx)
        # print("module:", modules[m_idx])
        h = hs[-1]
        logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
        h = modules[m_idx](h, temb)
        m_idx += 1
        # print("m_idx", m_idx)
        # print("module:", modules[m_idx])
        logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
        h = modules[m_idx](h)
        m_idx += 1
        # print("m_idx", m_idx)
        # print("module:", modules[m_idx])
        logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
        h = modules[m_idx](h, temb)
        m_idx += 1
        # print("m_idx", m_idx)
        # print("module:", modules[m_idx])
        # print(h.shape)

        pyramid = None

        # Upsampling block
        
        logging.debug("------ UPSAMPLING BLOCK ------")
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        h = h.permute(0, 3, 2, 1)
        logging.debug("Module %d: %s, parameters %d", m_idx, modules[m_idx]._get_name(),  sum(p.numel() for p in modules[m_idx].parameters() if p.requires_grad))
        h = modules[m_idx](h)
        m_idx += 1
        h = h.permute(0, 3, 2, 1)
        assert m_idx == len(modules)
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        logging.debug(f"END SHAPE: {h.shape}")
        return h
