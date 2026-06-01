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


"""
Paste this file and run in this repo https://github.com/Sprunckt/acoustic-sfw/src
please take care to have an other virtual environnement than the one used for the rest of the project
take care to have the right paths for the samples and the pos_eigenmike.npz
"""

import matplotlib.pyplot as plt
import numpy as np
from src.simulation.utils import (multichannel_rir_to_vec, vec_to_rir, create_grid_spherical,
                                compare_arrays)
from src.sfw import TimeDomainSFW

sde = 'sde' # sde or ode
type = 'measure' # simul or measure
for i in range(1,102):
    measure = i
    file = f"./pos_eigenmike.npz" # generated rir
    with np.load(file) as res:
        pos_eigenmike = res['pos_Emic']  # shape (3, 32)

    # use the file corresponding to the measure you want to test
    file_path = f"/home2020/home/icube/jdpascal/code_jd/acoustic-sfw/samples_{sde}_{type}_2/sample_0_{measure-1}.npz"
    crop_start = 45  # number of samples to add at the beginning of the RIRs

    with np.load(file_path) as sample :

        # Extraction generated_rir
        generated_rir = sample['generated_rir']
        print("dimension de la rir générée", generated_rir.shape)
        generated_rir = np.pad(generated_rir, ((crop_start,0),(0,0)),'constant',constant_values = 0)


        perfect_rir = sample['perfect_rir']
        perfect_rir = np.squeeze(perfect_rir, axis=0)
        perfect_rir = np.squeeze(perfect_rir, axis=0)
        perfect_rir = np.pad(perfect_rir, ((crop_start,0),(0,0)),'constant',constant_values = 0)


        real_rir = sample['real_rir']
        real_rir = np.squeeze(real_rir, axis=0)
        real_rir = np.squeeze(real_rir, axis=0)
        real_rir = np.pad(real_rir, ((crop_start,0),(0,0)),'constant',constant_values = 0)

        verite = sample['verite'] # ground truth source positions from pyroomacoustics
        ordre = sample['ordre']   # order of the sources in the verite array

    print("verite IS", verite.shape,ordre.shape)

    # Scene Parameters

    freq_sampling = 16000  # Hz

    # Load the eigenmike32 spherical microphone array
    # Source: https://www.locata.lms.tf.fau.de/files/2020/01/Documentation_LOCATA_final_release_V1.pdf
    # mic_array = np.transpose(np.genfromtxt('data/eigenmike32_cartesian.csv', delimiter=', '))
    mic_array = pos_eigenmike # other option : extracted from the .npz file


    N = 256 + crop_start  # number of time samples
    M = 32
    measurements = np.empty((N * M))

    for i in range(M):
        measurements[i * N:(i + 1) * N] = generated_rir[:, i] *  0.60  # rescalling t=0, a=1
        
    # measurements_2 : to test/ compare with the ground truth rir generated using pyroomacoustics
    measurements_2 = np.empty((N * M))
    for i in range(M):
        measurements_2[i * N:(i + 1) * N] = perfect_rir[:, i]


    d = 3  # dimension of the problem
    mic_array = mic_array.T  # positions, shape (M, d)

    J = N * M

    plot_sources = False
    if plot_sources:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(src[:, 0], src[:, 1], src[:, 2])
        ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
        plt.show()

    # create a spherical grid
    grid, sph_grid, n_sph = create_grid_spherical(1, 12, 0.3, 15, 15)
    print("grid shape : ", grid.shape)

    s = TimeDomainSFW(measurements / np.max(measurements), mic_pos=mic_array, fs=freq_sampling, N=N, lam=1e-2,end_tol=0.05) #,deletion_tol=None,end_tol=None

    save_dB = False # set to True if you want to save the dB value between the generated rir and the perfect rir, for Modulation Error Ratio (MER) computation
    dB = 0
    if save_dB :
        dB = 10 * np.log(np.linalg.norm(perfect_rir) ** 2 / np.linalg.norm(generated_rir - perfect_rir)**2 )


    compare = False
    if compare :
        # create a second SFW object with the measurements_2
        s_2 = TimeDomainSFW(measurements_2 / np.max(measurements_2), mic_pos=mic_array, fs=freq_sampling, N=N, lam=1e-2)

    plot_rir = False
    if plot_rir:  # plot the simulated RIR and the RIR computed using the measure gamma and the real source positions
        m = 5
        compg = vec_to_rir(s.gamma(ampl, src), m, N) # if you want to plot the ground truth from pyroomacoustics ampl, src for the amplitutde and position of the sources
        plt.plot(np.arange(N)/freq_sampling, compg/np.max(compg), label='gamma')
        rir = vec_to_rir(measurements, m, N)
        plt.plot(np.arange(N)/freq_sampling, rir/np.max(rir), '-.', label='pyroom')
        plt.xlabel("t (s)"), plt.ylabel("p")
        plt.legend()
        plt.show()

    load = False
    if load:
        res = np.load(f'../mes_{measure}_.npz')
        x, a, r = res['x'], res['a'], res['rir']
    else:
        a, x = s.reconstruct(grid, 20, 1.1, 8,spike_merging=True) # , slide_opt=dict(method="slide_once") if you want to use slide once
        r = s.gamma(a, x)
        np.savez(f'./mes_{type}_{sde}_2/mes_{measure}_.npz', a=a, x=x, rir=r, verite=verite, ordre=ordre, dB = dB)
        if compare:
            a_2, x_2 = s_2.reconstruct(grid, 20)
            r_2 = s_2.gamma(a_2, x_2)
            np.savez(f'../mes_{measure}_2.npz', a=a_2, x=x_2, rir=r_2)

    plot_reconstr = False
    if plot_reconstr:
        m = 0

        r = vec_to_rir(r/np.max(r), m, N)

        plt.plot(np.arange(N)/freq_sampling, r, label='reconstruction')
        rir = vec_to_rir(measurements/np.max(measurements), m, N)
        plt.plot(np.arange(N)/freq_sampling, rir, '-.', label='pyroom')
        plt.legend()
        plt.show()

        ind, dist = compare_arrays(src, x)
        print("distances between real and predicted  sources : \n", dist)

