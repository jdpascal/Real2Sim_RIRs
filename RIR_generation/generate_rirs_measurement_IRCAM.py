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

import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path
import gzip


import numpy as np
import pyroomacoustics as pra
import pyroomacoustics.directivities.sofa as sf
from alive_progress import alive_bar
from pyroomacoustics.datasets import SOFADatabase
from pyroomacoustics.directivities import (
    MeasuredDirectivityFile,
    Rotation3D,
)

import function as fun

# Constants
num_room = 1
positions_per_room = 10
# distance_src_mics = 1.53
dist_walls = 0.9
max_order_ism = 10
delay = 78
limit = 1024
crop_start = 85

# Coefficient of absorption more real, per octave band, per walls
abs_coeffs_lower_bound = np.array(
    [
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.05, 0.15, 0.25, 0.30],
        [0.01, 0.15, 0.40, 0.40, 0.40, 0.30],
    ]
)

abs_coeffs_upper_bound = np.array(
    [
        [0.50, 0.50, 0.30, 0.12, 0.12, 0.12],
        [0.50, 0.50, 0.30, 0.12, 0.12, 0.12],
        [0.50, 0.50, 0.30, 0.12, 0.12, 0.12],
        [0.50, 0.50, 0.30, 0.12, 0.12, 0.12],
        [0.20, 0.30, 0.50, 0.60, 0.75, 0.80],
        [0.70, 1.00, 1.00, 1.00, 1.00, 1.00],
    ]
)

center_freqs = [125, 250, 500, 1000, 2000, 4000, 8000]


# def calculate_rirs_for_config(
#     list_dir: list,
#     genelec8020 ,
#     room_dim: list[float],
#     pos_src: np.ndarray,
#     uncertainty_pos,
#     pos_mics: np.ndarray,
#     pos_eigenmike,
#     output_file_path: Path,
#     seed,
# ):
#     """
#     Calculates RIRs for the given configuration and writes result to an output JSON file.
#     """
#     # Orientation, test with source turn his back to the mics
#     cartesian_coords = pos_mics - pos_src
#     r, theta, phi = fun.cartesian_to_spherical(cartesian_coords) # remove the "-" to have the source facing the mics
#     pos_src += uncertainty_pos
#     uncertainty_angle = np.random.rand(2) * 10 - 5      # Random uncertainty in the angle of 5 degrees
#     theta += uncertainty_angle[0]
#     phi += uncertainty_angle[1]
    
#     orientation = Rotation3D([ -theta, phi ], "yz", degrees=True)
#     theta_mic, phi_mic = fun.random_angles()
#     orientation_mic = Rotation3D([-theta_mic, phi_mic], "yz", degrees=True)
#     # dir = DirectionVector(theta, phi)


#     # Pick random coefficient for absorption but realistic
#     P = np.random.uniform(abs_coeffs_lower_bound, abs_coeffs_upper_bound)
#     coin_flip = np.random.rand(6, 1) > 0.5
#     band_abs_profiles = coin_flip * P[np.random.randint(0, 5, 6), :] + (
#         ~coin_flip
#     ) * np.random.uniform(0.01, 0.12, (6, 6))
#     band_abs_profiles = np.concatenate(
#         (band_abs_profiles, band_abs_profiles[:, -1].reshape(6, -1)), axis=1
#     )
#     full_materials = pra.make_materials(
#         ceiling={
#             "description": "5",
#             "coeffs": (band_abs_profiles[5, :]).tolist(),
#             "center_freqs": center_freqs,
#         },
#         floor={
#             "description": "4",
#             "coeffs": (band_abs_profiles[4, :]).tolist(),
#             "center_freqs": center_freqs,
#         },
#         east={
#             "description": "1",
#             "coeffs": (band_abs_profiles[1, :]).tolist(),
#             "center_freqs": center_freqs,
#         },
#         west={
#             "description": "0",
#             "coeffs": (band_abs_profiles[0, :]).tolist(),
#             "center_freqs": center_freqs,
#         },
#         north={
#             "description": "3",
#             "coeffs": (band_abs_profiles[3, :]).tolist(),
#             "center_freqs": center_freqs,
#         },
#         south={
#             "description": "2",
#             "coeffs": (band_abs_profiles[2, :]).tolist(),
#             "center_freqs": center_freqs,
#         },
#     )
#     full_materials_mean = pra.make_materials(
#         ceiling={
#             "description": "5",
#             "coeffs": [np.mean(band_abs_profiles[5, :])],
#         },
#         floor={
#             "description": "4",
#             "coeffs": [np.mean(band_abs_profiles[4, :])],
#         },
#         east={
#             "description": "1",
#             "coeffs": [np.mean(band_abs_profiles[1, :])],
#         },
#         west={
#             "description": "0",
#             "coeffs": [np.mean(band_abs_profiles[0, :])],
#         },
#         north={
#             "description": "3",
#             "coeffs": [np.mean(band_abs_profiles[3, :])],
#         },
#         south={
#             "description": "2",
#             "coeffs": [np.mean(band_abs_profiles[2, :])],
#         },
#     )

#     # Create the real room with real micro and source
#     room_real = pra.ShoeBox(
#         room_dim,
#         fs=16000,
#         max_order=max_order_ism,
#         materials=full_materials,
#         air_absorption=True,
#         ray_tracing=False,
#         min_phase=True,
#         use_rand_ism=True,
#         max_rand_disp=0.2,
#         flip_walls=True,
#     )

#     # Add source and microphone
#     genelec8020.set_orientation(orientation)
#     room_real.add_source(pos_src, directivity=genelec8020)
#     # Get the directivity objects from the files and add mics
#     for j in range(32):
#         (list_dir[j]).set_orientation(orientation_mic)
#     room_real.add_microphone_array(
#         (np.zeros((32,3)) + pos_mics).T, directivity=list_dir
#     )

#     # Create the "perfect" room with omnidirectional micro and source
#     room_perfect = pra.ShoeBox(
#         room_dim,
#         fs=16000,
#         max_order=max_order_ism,
#         materials=full_materials_mean,
#         air_absorption=True,
#         ray_tracing=False,
#         min_phase=True,
#         use_rand_ism=True,
#         max_rand_disp=0.2,
#         flip_walls=False,
#     )

#     # Add source and microphone omnidirectionnal
#     room_perfect.add_source(pos_src)
#     # Turn the position of the mics to be the same as the Eigenmike turned
#     list_pos = []
#     for i in range(32):
#         new_pos = orientation_mic.rotate(pos_eigenmike.T[i])
#         list_pos.append(np.array(new_pos))
#     room_perfect.add_microphone_array((list_pos + pos_mics).T) 
#     # room_perfect.add_microphone_array((pos_eigenmike.T + pos_mics).T)

#     # Compute the RIR
#     np.random.seed(seed)  # For reproducibility
#     room_real.compute_rir()
#     # logging.info("RIR computed for the real room.")
#     np.random.seed(seed)  # For reproducibility
#     room_perfect.compute_rir()
#     # logging.info("RIR computed for the perfect room.")
#     rir_real = room_real.rir
#     rir_perfect = room_perfect.rir

#     # Reshape and transform as array instead of list of list
#     max_perfect = np.max(np.array([(rir_perfect[i][0]).shape[0] for i in range(32)]))
#     test_perfect = np.pad(
#         np.array(rir_perfect[0][0]),
#         (0, max_perfect - len(rir_perfect[0][0])),
#         "constant",
#         constant_values=[0, 0],
#     )

#     for i in range(1, 32):
#         test_perfect = np.concatenate(
#             (
#                 test_perfect,
#                 np.pad(
#                     np.array(rir_perfect[i][0]),
#                     (0, max_perfect - len(rir_perfect[i][0])),
#                     "constant",
#                     constant_values=[0, 0],
#                 ),
#             )
#         )

#     test_perfect = test_perfect.reshape(32, max_perfect)
#     max_real = np.max(np.array([(rir_real[i][0]).shape[0] for i in range(32)]))
#     test_real = np.pad(
#         np.array(rir_real[0][0]),
#         (0, max_real - len(rir_real[0][0])),
#         "constant",
#         constant_values=[0, 0],
#     )

#     for i in range(1, 32):
#         test_real = np.concatenate(
#             (
#                 test_real,
#                 np.pad(
#                     np.array(rir_real[i][0]),
#                     (0, max_real - len(rir_real[i][0])),
#                     "constant",
#                     constant_values=[0, 0],
#                 ),
#             )
#         )

#     test_real = test_real.reshape(32, max_real)

#     real_rir = test_real[:, crop_start + delay : crop_start + limit + delay] #/ np.max(
#         # test_real[:, crop_start + delay : crop_start + limit + delay], axis=1
#     # ).reshape(-1, 1)
#     real_rir = real_rir / np.max( np.abs(real_rir) )
#     perfect_rir = test_perfect[:, crop_start : crop_start + limit]# / np.max(
#     #     test_perfect[:, crop_start : crop_start + limit], axis=1
#     # ).reshape(-1, 1)
#     perfect_rir = perfect_rir / np.max( np.abs(perfect_rir) )
#     for s, src in enumerate(room_perfect.sources):
#         order = src.orders
#         src_image = (src.images)
#         list_src = []
#         list_order = []
#         for i in range(src_image.shape[1]):
#             if np.linalg.norm(src_image[: , i] - pos_mics) < 7.4:
#                 list_src.append(src_image[:, i] - pos_mics)
#                 list_order.append(order[i])

#         list_src = np.array(list_src)
#         list_order = np.array(list_order)
#         # simplifiée puis algo de tom
#         idx_sorted = np.argsort(np.linalg.norm(list_src,axis=1))
#         x_sorted = list_src[idx_sorted]
#         list_src = x_sorted
#         x_sorted_ord = list_order[idx_sorted]
#         list_order = x_sorted_ord
#     # Store all calculated values for this configuration in a dict
#     # We need to use tolist() here to convert from numpy arrays to python arrays that can be serialized to json
#     room_data = {
#         "real_rir": real_rir.tolist(),
#         "perfect_rir": perfect_rir.tolist(),
#         "max_order_ism": max_order_ism,
#         "geometry": {
#             "room_dim": room_dim,
#             "pos_src": pos_src.tolist(),
#             "pos_mics": pos_mics.tolist(),
#         },
#         "abs_coeffs": band_abs_profiles.tolist(),
#         'ground_truth' : list_src.tolist(),#np.sort(np.linalg.norm(list_src, axis=1)),
#         'oreder' : list_order.tolist(),
#     }
#     # Write the contents of this dict to a file named with the number of the room and
#     # current src/rcv position iteration
#     with gzip.open(output_file_path, "wt", encoding="utf-8") as f:
#         json.dump(room_data, f)


# def main():
#     workdir = "./dataset_iwaenc/"
#     os.makedirs(workdir, exist_ok=True)
    
#     # --- Configuration du logger pour écrire dans un fichier ---
#     log_file = os.path.join(workdir, "stdout.txt")
#     gfile_stream = open(log_file, "w", encoding="utf-8")
#     formatter = logging.Formatter(
#         "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
#     )
#     handler = logging.StreamHandler(gfile_stream)
#     handler.setFormatter(formatter)
#     logger = logging.getLogger()
#     # logger.addHandler(handler)
#     # Add another handler to also log to stdout
#     handler = logging.StreamHandler(sys.stdout)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)
    
#     # logger.info("Opening SOFA database...")
#     db = SOFADatabase()
#     # logger.info("...Done")
#     # db.list()
#     # print(db['EM32_Directivity'])
#     # list = download_sofa_files()

#     logger.info("Reading data from SOFA database...")
#     logger.info("   Reading EigenMike data...")
#     # Reads the file containing the Eigenmike's directivity measurements
#     eigenmike = MeasuredDirectivityFile("EM32_Directivity", fs=16000, interp_order=18)
#     logger.info("   ...Done")

#     # Reads the file containing Genelec 8020 's directivity measurements
#     logger.info("   Reading Genelec data...")
#     src_dir = MeasuredDirectivityFile(
#         "LSPs_HATS_GuitarCabinets_Akustikmessplatz", fs=16000, interp_order=18
#     )
#     genelec8020 = src_dir.get_source_directivity(
#         "Genelec_8020", orientation=Rotation3D([0, 0], "yz", degrees=True)
#     )
#     # logger.info("   ...Done")
    
#     # logger.info("   Reading Eigenmike directivity...")
#     list_dir = []
#     for j in range(32):
#         # logger.info("       %d / 32...", j + 1)
#         dir_obj_Emic = eigenmike.get_mic_directivity(
#             f"EM_32_{j}", orientation=Rotation3D([0, 0], "yz", degrees=True)
#         )
#         list_dir.append(dir_obj_Emic)
#     # logger.info("   ...Done")

#     # logger.info("   Reading Eigenmike position...")
#     path = db["EM32_Directivity"].path
#     files = sf.open_sofa_file(path)
#     pos_eigenmike = files[3]
#     # logger.info("   ...Done")
#     # logger.info("...Done")

#     logger.info("Starting generating room configurations")

#     configurations = []
#     for room_index in range(0, num_room):
#         logger.info("   Generate room config n°%d...", room_index)
#         # Dimensions of the room
#         Dx, Dy, Dz = fun.generate_random_room_dimensions(
#             min_size_x=3.5, max_size_x=10.0, min_size_y=3.5, max_size_y=10.0, min_size_z=2.0, max_size_z=4.5
#         )
#         room_dim = [Dx, Dy, Dz]
#         # Generate #positions_per_room measurement_annotations in the room
#         for position_index in range(positions_per_room):
#             # logger.info("       Position %d / %d", position_index + 1, positions_per_room)
#             # Generate 2 random points in the room, with constraints on location
#             approx = fun.approximation_distance(0.1)
#             pos_src, pos_mics = fun.generate_random_points(
#                 Dx, Dy, Dz, distance_src_mics , dist_walls
#             )
#             uncertainty_pos = (np.random.rand(3) * 2 - 1) / 10     # Random uncertainty in the position of 10 cm
#             configurations.append(
#                 (
#                     list_dir,
#                     genelec8020,
#                     room_dim,
#                     pos_src,
#                     uncertainty_pos,
#                     pos_mics,
#                     pos_eigenmike,
#                     Path(workdir) / f"room_{room_index}_{position_index}.json.gz",
#                     np.random.randint(0, 1000000),
#                 )
#             )
#     #     logger.info("   ...Done")
            
#     # logger.info("Done generating room configs")

#     # # Run the function in parallel and wait for all processes to complete
#     # logger.info("Start computing RIRs...")
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [
#             executor.submit(calculate_rirs_for_config, *config)
#             for config in configurations
#         ]
#         # Create a progress bar that will fill up as new rirs are calculated
#         with alive_bar(num_room * positions_per_room) as bar:
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     future.result()  # This will raise an exception if the future raised one
#                     bar()  # pylint:disable=not-callable
#                 except Exception as e:
#                     logger.error(f"An error occurred: {e}")

#     # logger.info("Done !")
#     # Fermeture du fichier de log
#     gfile_stream.close()


# if __name__ == "__main__":
#     main()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pyroomacoustics as pra
# from pyroomacoustics.datasets import SOFADatabase , download_sofa_files
# import function as fun
# import pyroomacoustics.directivities.sofa as sf
# import imageio
# from pyroomacoustics.directivities import (
#     Cardioid,
#     DirectionVector,
#     FigureEight,
#     MeasuredDirectivityFile,
#     Rotation3D,
# )

# import os
# import logging
# import json
# import numpy as np
# import sofa
# import gzip

db = SOFADatabase()

list = download_sofa_files()

import scipy.io as sio
from scipy.signal import resample_poly
liste_des_orders = []
path = db['EM32_Directivity'].path
files = sf.open_sofa_file( path )

pos_eigenmike = files[3]
#Reads the file containing Genelec 8020 's directivity measurements
src_dir = MeasuredDirectivityFile("LS_directivity_Calibrated_GENELEC_8030B", fs=16000,interp_order=18)
genelec8020 = src_dir.get_source_directivity(0, orientation=Rotation3D([ 0, 0], "yz", degrees=True))
# Reads the file containing the Eigenmike's directivity measurements
eigenmike = MeasuredDirectivityFile("EM32_Directivity", fs=16000,interp_order=18)

# # Coefficient of absorption more real, per octave band, per walls

# abs_coeffs_lower_bound = np.array([[0.01,0.01,0.01,0.01,0.01,0.01],
#     [0.01,0.01,0.01,0.01,0.01,0.01],
#     [0.01,0.01,0.01,0.01,0.01,0.01],
#     [0.01,0.01,0.01,0.01,0.01,0.01],
#     [0.01,0.01,0.05,0.15,0.25,0.30],
#     [0.01,0.15,0.40,0.40,0.40,0.30]])

# abs_coeffs_upper_bound = np.array([[0.50,0.50,0.30,0.12,0.12,0.12],
#     [0.50,0.50,0.30,0.12,0.12,0.12],
#     [0.50,0.50,0.30,0.12,0.12,0.12],
#     [0.50,0.50,0.30,0.12,0.12,0.12],
#     [0.20,0.30,0.50,0.60,0.75,0.80],
#     [0.70,1.00,1.00,1.00,1.00,1.00]])

# center_freqs = [125,250,500,1000,2000,4000,8000]


workdir = './dataset_ircam'
os.makedirs(workdir, exist_ok=True)

# # --- Configuration du logger pour écrire dans un fichier ---
log_file = os.path.join(workdir, 'stdout.txt')
gfile_stream = open(log_file, 'w')
# handler = logging.StreamHandler(gfile_stream)
# formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
# handler.setFormatter(formatter)
# logger = logging.getLogger()
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

measurement_annotation = {
    'room_dim': [[9.53, 4.72, 2.78],
                [9.53, 4.72, 2.78],
                [9.53, 4.72, 2.78],
                [9.53, 4.72, 2.78],
                [9.53, 4.72, 2.78],
                [9.53, 4.72, 2.78],
                [9.53, 4.72, 2.39],
                [9.53, 4.72, 2.39],
                [9.53, 4.72, 2.39],
                [9.53, 4.72, 2.39],
                [9.53, 4.72, 2.39],
                [9.53, 4.72, 2.39]],

    'pos_src': [np.array([4.11, 2.1, 1.71]),
                np.array([3.64, 2.88, 1.71]),
                np.array([2.94, 3.45 , 1.71]),
                np.array([1.96, 3.08 , 1.71]),
                np.array([3.11, 3.29 , 1.71]),
                np.array([3.76, 3.3 , 1.71]),
                np.array([7.32, 2.87 , 1.35]),
                np.array([7.51, 2.06 , 1.35]),
                np.array([9.53 - 1.69 , 1.71 , 1.35]),
                np.array([9.53 - 3.27,2.14 , 1.35]),
                np.array([9.53 - 3.27,2.14 , 1.35]),
                np.array([5.00, 1.58 , 1.35])],
    'pos_mics': [np.array([2.63, 1.32, 1.71]),
                np.array([2.31, 1.91, 1.71]),
                np.array([2.57, 1.84, 1.71]),
                np.array([2.96, 1.77, 1.71]),
                np.array([3.13, 1.63, 1.71]),
                np.array([2.11, 3.27, 1.71]),
                np.array([5.81, 2.12, 1.35]),
                np.array([6.3, 3.2, 1.35]),
                np.array([7.36, 3.25, 1.35]),
                np.array([9.53 - 1.77, 2.83, 1.35]),
                np.array([9.53 - 1.77, 2.83, 1.35]),
                np.array([5.03, 3.24, 1.35])],
}

measurement_annotation_2 = {   ### as we put the devices on a rolling trolley, there is some reflections on the trolley that should appear in the ground truth, so we compute another expecting location of the IS to complete the ground truth
    'room_dim': [[9.53, 4.72, 2.48],
                [9.53, 4.72, 2.48],
                [9.53, 4.72, 2.48],
                [9.53, 4.72, 2.48],
                [9.53, 4.72, 2.48],
                [9.53, 4.72, 2.48],
                [9.53, 4.72, 2.09],
                [9.53, 4.72, 2.09],
                [9.53, 4.72, 2.09],
                [9.53, 4.72, 2.09],
                [9.53, 4.72, 2.39],
                [9.53, 4.72, 2.39]],

    'pos_src': [np.array([4.11, 2.1, 1.41]),
                np.array([3.64, 2.88, 1.41]),
                np.array([2.94, 3.45 , 1.41]),
                np.array([1.96, 3.08 , 1.41]),
                np.array([3.11, 3.29 , 1.41]),
                np.array([3.76, 3.3 , 1.41]),
                np.array([7.32, 2.87 , 1.05]),
                np.array([7.51, 2.06 , 1.05]),
                np.array([9.53 - 1.69 , 1.71 , 1.05]),
                np.array([9.53 - 3.27,2.14 , 1.05]),
                np.array([9.53 - 3.27,2.14 , 1.05]),
                np.array([5.00, 1.58 , 1.05])],
    'pos_mics': [np.array([2.63, 1.32, 1.41]),
                 np.array([2.31, 1.91, 1.41]),
                 np.array([2.57, 1.84, 1.41]),
                 np.array([2.96, 1.77, 1.41]),
                 np.array([3.13, 1.63, 1.41]),
                 np.array([2.11, 3.27, 1.41]),
                 np.array([5.81, 2.12, 1.05]),
                 np.array([6.3, 3.2, 1.05]),
                 np.array([7.36, 3.25, 1.05]),
                 np.array([9.53 - 1.77, 2.83, 1.05]),
                 np.array([9.53 - 1.77, 2.83, 1.35]),
                 np.array([5.03, 3.24, 1.35])],
}
# logger.info("Début de la génération des données.")
room  = 0
# max_dist is the maximum distance of visible IS in the 256 first samples (more the 45 first samples we already remove)
max_dist = (256 + 45) * 343 / 16000
# this threshold on radial distance will be applied on IS matching 
threshold_r = 1.0   


for measure_idx in range(positions_per_room):
    test = sio.loadmat(f'measurement_annotations/LGS_M{measure_idx:02d}_S{measure_idx:02d}.mat')


    # Paramètres de rééchantillonnage
    fs_in = 48000
    fs_out = 16000
    down_factor = fs_in // fs_out  # ici, 3

    # Downsampling par canal avec resample_poly (plus qualitatif que slicing simple)
    audio_downsampled = resample_poly(test['ir_m'], up=1, down=down_factor, axis=1)


    max = np.max(audio_downsampled[:,783 :783 + 1024])  # it is hardcoded, we aligned the RIRS with simulation because we want to calibrate the microphones

    data_dict = {
        'real_rir': [],         # for the real room
        'perfect_rir': [],         # for the perfect room
        'geometry': { 
            'room_dim': [],     # for the dimensions of the room
            'pos_src': [],      # for the position of the source
            'pos_mics': [],     # for the position of the microphones
        },  # for the geometry of the room
            'abs_coeffs': [],   # for the absorption coefficients of the walls
            'max_order_ism': max_order_ism,  # for the maximum order of the ISM
    }
    # print(room)
    # Dimensions of the room
    room_dim = measurement_annotation['room_dim'][measure_idx-1]
    data_dict['geometry']['room_dim'].append(np.array(room_dim))
    # Generate 2 random points in the room, with constraints on location
    pos_src = measurement_annotation['pos_src'][measure_idx-1]
    pos_mics = measurement_annotation['pos_mics'][measure_idx-1]
    data_dict['geometry']['pos_src'].append(pos_src)
    data_dict['geometry']['pos_mics'].append(pos_mics)

    # Orientation
    cartesian_coords = pos_mics - pos_src 
    r, theta, phi = fun.cartesian_to_spherical(cartesian_coords)
    orientation = Rotation3D([-theta , phi ], "yz", degrees=True)
    theta_mic, phi_mic = 0,0
    orientation_mics = Rotation3D([theta_mic, phi_mic], "yz", degrees=True)


    # Pick random coefficient for absorption but realistic
    P = np.random.uniform(abs_coeffs_lower_bound,abs_coeffs_upper_bound)
    coin_flip = (np.random.rand(6,1) > 0.5)
    band_abs_profiles= coin_flip * P[np.random.randint(0,5,6),:]+(~coin_flip)*np.random.uniform(0.01,0.12,(6,6))
    band_abs_profiles = np.concatenate((band_abs_profiles,band_abs_profiles[:,-1].reshape(6,-1)),axis = 1)
    full_materials = pra.make_materials(
        ceiling={"description":'5',"coeffs" :((band_abs_profiles[5,:])).tolist(),"center_freqs" :center_freqs},
        floor={"description":'4',"coeffs" :(band_abs_profiles[4,:]).tolist(),"center_freqs" :center_freqs},
        east={"description":'1',"coeffs" :(band_abs_profiles[1,:]).tolist(),"center_freqs" :center_freqs},
        west={"description":'0',"coeffs" :(band_abs_profiles[0,:]).tolist(),"center_freqs" :center_freqs},
        north={"description":'3',"coeffs" :(band_abs_profiles[3,:]).tolist(),"center_freqs" :center_freqs},
        south={"description":'2',"coeffs" :(band_abs_profiles[2,:]).tolist(),"center_freqs" :center_freqs}
    )
    data_dict['abs_coeffs'].append(band_abs_profiles)
    
    # Create the real room with real micro and source

    room_real = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=max_order_ism,
        materials=full_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=True,
        flip_walls=True,  
    )

    # Add source and microphone 
    genelec8030 = src_dir.get_source_directivity(0, orientation=orientation)
    room_real.add_source(pos_src, directivity=genelec8030)   
    # Get the directivity objects from the files and add mics
    list_dir = []
    for j in range (32):
        dir_obj_Emic = eigenmike.get_mic_directivity(f"EM_32_{j}", orientation=orientation_mics)
        list_dir.append(dir_obj_Emic)
    room_real.add_microphone_array(( np.zeros((32,3)) + pos_mics).T, directivity=list_dir) # np.zeros((32,3)) pos_eigenmike.T

    # Create the "perfect" room with omnidirectional micro and source
    room_perfect = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=max_order_ism,
        materials=full_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=True,
    )
    room_perfect_2 = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=max_order_ism,
        materials=full_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=True,
    )
    room_dim_2 = measurement_annotation_2['room_dim'][measure_idx-1]
    # Generate 2 random points in the room, with constraints on location
    pos_src_2 = measurement_annotation_2['pos_src'][measure_idx-1]
    pos_mics_2 = measurement_annotation_2['pos_mics'][measure_idx-1]


    # Add source and microphone omnidirectionnal
    room_perfect.add_source(pos_src)   
    list_pos = []
    for i in range(32):
        new_pos = orientation_mics.rotate(pos_eigenmike.T[i])
        list_pos.append(np.array(new_pos))
        # print(np.sqrt(np.sum(np.array(new_pos)**2,axis=0)))
    room_perfect.add_microphone_array((list_pos + pos_mics).T) 
    # room_real.add_microphone_array(( list_pos + pos_mics).T)#, directivity=list_dir) # np.zeros((32,3)) pos_eigenmike.T

    room_perfect_2.add_source(pos_src_2)   
    room_perfect_2.add_microphone_array((list_pos + pos_mics_2).T) 

    # Compute the RIR
    room_real.compute_rir()
    room_perfect.compute_rir()
    room_perfect_2.compute_rir()
    rir_real = room_real.rir
    rir_perfect = room_perfect.rir
    rir_perfect_2 = room_perfect_2.rir

    
    # Reshape and transform as array instead of list of list
    max_perfect = np.max(np.array([(rir_perfect[i][0]).shape[0] for i in range(32)]))
    test_perfect = np.pad(np.array(rir_perfect[0][0]),(0,max_perfect-rir_perfect[0][0].shape[0]),'constant',constant_values=[0,0])

    for i in range(1, 32):
        test_perfect = np.concatenate((test_perfect, np.pad(np.array(rir_perfect[i][0]), (0, max_perfect - (rir_perfect[i][0]).shape[0]), 'constant', constant_values=[0, 0])))

    test_perfect = test_perfect.reshape(32, max_perfect)

    # max_perfect_2 = np.max(np.array([(rir_perfect_2[i][0]).shape[0] for i in range(32)]))
    # test_perfect_2 = np.pad(np.array(rir_perfect_2[0][0]),(0,max_perfect_2-rir_perfect_2[0][0].shape[0]),'constant',constant_values=[0,0])

    # for i in range(1, 32):
    #     test_perfect_2 = np.concatenate((test_perfect_2, np.pad(np.array(rir_perfect_2[i][0]), (0, max_perfect_2 - (rir_perfect_2[i][0]).shape[0]), 'constant', constant_values=[0, 0])))

    # test_perfect_2 = test_perfect.reshape(32, max_perfect_2)
    
    max_real = np.max(np.array([(rir_real[i][0]).shape[0] for i in range(32)]))
    test_real = np.pad(np.array(rir_real[0][0]), (0, max_real - (rir_real[0][0]).shape[0]), 'constant', constant_values=[0, 0])

    for i in range(1, 32):
        test_real = np.concatenate((test_real, np.pad(np.array(rir_real[i][0]), (0, max_real - (rir_real[i][0]).shape[0]), 'constant', constant_values=[0, 0])))

    test_real = test_real.reshape(32, max_real)

    # Normalize the RIR and put in the dictionnary
    data_dict['measure'] = ((-audio_downsampled[ : , 783 : 783 + limit ] / max))
    data_dict['real_rir'] = (test_real[ : , crop_start + delay : crop_start + limit + delay] / np.max(test_real[ : , crop_start + delay : crop_start + limit + delay]).reshape(-1,1))
    data_dict['perfect_rir'] = (test_perfect[ : , crop_start : crop_start + limit ] / np.max(test_perfect[ : , crop_start : crop_start + limit ]).reshape(-1,1))
    

    # Enregistrement du dictionnaire dans un fichier JSON
    json_file = os.path.join(workdir, f'room_{room}_{measure_idx}.json.gz')
    # with open(json_file, 'w') as f:
    for s, src in enumerate(room_perfect.sources):
        order = src.orders
        src_image = (src.images)
        list_src = []
        list_order = []
        for i in range(src_image.shape[1]):
            if np.linalg.norm(src_image[: , i] - pos_mics) < max_dist + threshold_r :
                list_src.append(src_image[:, i] - pos_mics)
                list_order.append(order[i])

        list_src = np.array(list_src)
        list_order = np.array(list_order)
        # simplifiée puis algo de tom
        idx_sorted = np.argsort(np.linalg.norm(list_src,axis=1))
        x_sorted = list_src[idx_sorted]
        list_src = x_sorted
        x_sorted_ord = list_order[idx_sorted]
        list_order = x_sorted_ord

    for s, src in enumerate(room_perfect_2.sources):
        order = src.orders
        src_image = (src.images)
        list_src_2 = []
        list_order_2 = []
        for i in range(src_image.shape[1]):
            if np.linalg.norm(src_image[: , i] - pos_mics_2) < max_dist + threshold_r :
                list_src_2.append(src_image[:, i] - pos_mics_2)
                list_order_2.append(order[i])

        list_src_2 = np.array(list_src_2)
        list_order_2 = np.array(list_order_2)
        # simplifiée puis algo de tom
        idx_sorted = np.argsort(np.linalg.norm(list_src_2,axis=1))
        x_sorted = list_src_2[idx_sorted]
        list_src_2 = x_sorted
        x_sorted_ord = list_order_2[idx_sorted]
        list_order_2 = x_sorted_ord

    with gzip.open(json_file, "wt", encoding="utf-8") as f:
        data_dict_json = {
            'real_rir': [rir.tolist() for rir in data_dict['real_rir']],
            'perfect_rir': [rir.tolist() for rir in data_dict['perfect_rir']],
            'measurement_rir': [rir.tolist() for rir in data_dict['measure']],
            'geometry': {
                'room_dim': [dim.tolist() for dim in data_dict['geometry']['room_dim']],
                'pos_src': [pos.tolist() for pos in data_dict['geometry']['pos_src']],
                'pos_mics': [pos.tolist() for pos in data_dict['geometry']['pos_mics']],
            },
            'abs_coeffs': [coeff.tolist() for coeff in data_dict['abs_coeffs']],
            'max_order_ism': data_dict['max_order_ism'],
            'ground_truth' : list_src.tolist(),     #then we will merge ground_truth and ground_truth_
            'ground_truth_2' : list_src_2.tolist(),
            'order' : list_order.tolist(),
            'order_2' : list_order_2.tolist(),
        }
        json.dump(data_dict_json, f)

    # Fermeture du fichier de log
    gfile_stream.close()

