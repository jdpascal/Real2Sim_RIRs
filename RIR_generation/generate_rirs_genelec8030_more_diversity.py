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
    MeasuredDirectivity,
    Rotation3D,
)

import function as fun

# Constants
num_room = 150
positions_per_room = 3
distance_src_mics = 1.63
dist_walls = 1
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


def calculate_rirs_for_config(
    list_dir: list,
    genelec8030 ,
    room_dim: list[float],
    pos_src: np.ndarray,
    uncertainty_pos,
    pos_mics: np.ndarray,
    pos_eigenmike,
    output_file_path: Path,
    seed,
):
    """
    Calculates RIRs for the given configuration and writes result to an output JSON file.
    """
    logging.info(f"Calculating RIRs for configuration: {output_file_path}")
    # Orientation, test with source turn his back to the mics
    cartesian_coords = pos_mics - pos_src
    r, theta, phi = fun.cartesian_to_spherical(cartesian_coords)
    pos_src += uncertainty_pos
    uncertainty_angle = np.random.rand(2) * 10 - 5      # Random uncertainty in the angle of 5 degrees
    theta += uncertainty_angle[0]
    phi += uncertainty_angle[1]

    # theta_src, phi_src = fun.random_angles()
    orientation = Rotation3D([ -theta, phi ], "yz", degrees=True)
    theta_mic, phi_mic = fun.random_angles()
    orientation_mic = Rotation3D([-theta_mic, phi_mic], "yz", degrees=True)

    # Pick random coefficient for absorption but realistic
    P = np.random.uniform(abs_coeffs_lower_bound, abs_coeffs_upper_bound)
    coin_flip = np.random.rand(6, 1) > 0.5
    band_abs_profiles = coin_flip * P[np.random.randint(0, 5, 6), :] + (
        ~coin_flip
    ) * np.random.uniform(0.01, 0.12, (6, 6))
    band_abs_profiles = np.concatenate(
        (band_abs_profiles, band_abs_profiles[:, -1].reshape(6, -1)), axis=1
    )
    full_materials = pra.make_materials(
        ceiling={
            "description": "5",
            "coeffs": (band_abs_profiles[5, :]).tolist(),
            "center_freqs": center_freqs,
        },
        floor={
            "description": "4",
            "coeffs": (band_abs_profiles[4, :]).tolist(),
            "center_freqs": center_freqs,
        },
        east={
            "description": "1",
            "coeffs": (band_abs_profiles[1, :]).tolist(),
            "center_freqs": center_freqs,
        },
        west={
            "description": "0",
            "coeffs": (band_abs_profiles[0, :]).tolist(),
            "center_freqs": center_freqs,
        },
        north={
            "description": "3",
            "coeffs": (band_abs_profiles[3, :]).tolist(),
            "center_freqs": center_freqs,
        },
        south={
            "description": "2",
            "coeffs": (band_abs_profiles[2, :]).tolist(),
            "center_freqs": center_freqs,
        },
    )
    full_materials_mean = pra.make_materials(
        ceiling={
            "description": "5",
            "coeffs": [np.mean(band_abs_profiles[5, :])],
        },
        floor={
            "description": "4",
            "coeffs": [np.mean(band_abs_profiles[4, :])],
        },
        east={
            "description": "1",
            "coeffs": [np.mean(band_abs_profiles[1, :])],
        },
        west={
            "description": "0",
            "coeffs": [np.mean(band_abs_profiles[0, :])],
        },
        north={
            "description": "3",
            "coeffs": [np.mean(band_abs_profiles[3, :])],
        },
        south={
            "description": "2",
            "coeffs": [np.mean(band_abs_profiles[2, :])],
        },
    )

    # Create the real room with real micro and source
    room_real = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=max_order_ism,
        materials=full_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=True,
        use_rand_ism=True,
        max_rand_disp=0.2,
        flip_walls=True,
    )

    # Add source and microphone
    genelec8030.set_orientation(orientation)
    room_real.add_source(pos_src, directivity=genelec8030)
    # Get the directivity objects from the files and add mics
    for j in range(32):
        (list_dir[j]).set_orientation(orientation_mic)
    room_real.add_microphone_array(
        (np.zeros((32,3)) + pos_mics).T, directivity=list_dir
    )  # , directivity=list_dir

    # Create the "perfect" room with omnidirectional micro and source
    room_perfect = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=max_order_ism,
        materials=full_materials_mean,
        air_absorption=True,
        ray_tracing=False,
        min_phase=True,
        use_rand_ism=True,
        max_rand_disp=0.2,
        flip_walls=False,
    )

    # Add source and microphone omnidirectionnal
    room_perfect.add_source(pos_src)
    # Turn the position of the mics to be the same as the Eigenmike turned
    list_pos = []
    for i in range(32):
        new_pos = orientation_mic.rotate(pos_eigenmike.T[i])
        list_pos.append(np.array(new_pos))
    room_perfect.add_microphone_array((list_pos + pos_mics).T) 
    # room_perfect.add_microphone_array((pos_eigenmike.T + pos_mics).T)
    # logging.info(
    #     f"Room created with dimensions: {room_dim}, source position: {pos_src}, microphone positions: {pos_mics}"
    # )

    # Compute the RIR
    np.random.seed(seed)  # For reproducibility
    room_real.compute_rir()
    # logging.info("RIR computed for the real room.")
    np.random.seed(seed)  # For reproducibility
    room_perfect.compute_rir()
    # logging.info("RIR computed for the perfect room.")
    rir_real = room_real.rir
    rir_perfect = room_perfect.rir

    # Reshape and transform as array instead of list of list
    max_perfect = np.max(np.array([(rir_perfect[i][0]).shape[0] for i in range(32)]))
    test_perfect = np.pad(
        np.array(rir_perfect[0][0]),
        (0, max_perfect - len(rir_perfect[0][0])),
        "constant",
        constant_values=[0, 0],
    )

    for i in range(1, 32):
        test_perfect = np.concatenate(
            (
                test_perfect,
                np.pad(
                    np.array(rir_perfect[i][0]),
                    (0, max_perfect - len(rir_perfect[i][0])),
                    "constant",
                    constant_values=[0, 0],
                ),
            )
        )

    test_perfect = test_perfect.reshape(32, max_perfect)
    max_real = np.max(np.array([(rir_real[i][0]).shape[0] for i in range(32)]))
    test_real = np.pad(
        np.array(rir_real[0][0]),
        (0, max_real - len(rir_real[0][0])),
        "constant",
        constant_values=[0, 0],
    )

    for i in range(1, 32):
        test_real = np.concatenate(
            (
                test_real,
                np.pad(
                    np.array(rir_real[i][0]),
                    (0, max_real - len(rir_real[i][0])),
                    "constant",
                    constant_values=[0, 0],
                ),
            )
        )

    test_real = test_real.reshape(32, max_real)

    # real_rir = np.array(rir_edit)
    real_rir = test_real[:, crop_start + delay : crop_start + limit + delay] #/ np.max( np.abs(
        # test_real[:, crop_start + delay : crop_start + limit + delay]))
    real_rir = real_rir / np.max( np.abs(real_rir) )
    perfect_rir = test_perfect[:, crop_start : crop_start + limit] #/ np.max( np.abs(
        # test_perfect[:, crop_start : crop_start + limit]))
    perfect_rir = perfect_rir / np.max( np.abs(perfect_rir) )
    for s, src in enumerate(room_perfect.sources):
        order = src.orders
        src_image = (src.images)
        list_src = []
        list_ordre = []
        for i in range(src_image.shape[1]):
            if np.linalg.norm(src_image[: , i] - pos_mics) < 7.4:
                list_src.append(src_image[:, i] - pos_mics)
                list_ordre.append(order[i])

        list_src = np.array(list_src)
        list_ordre = np.array(list_ordre)
        # simplifiée puis algo de tom
        idx_sorted = np.argsort(np.linalg.norm(list_src,axis=1))
        x_sorted = list_src[idx_sorted]
        list_src = x_sorted
        x_sorted_ord = list_ordre[idx_sorted]
        list_ordre = x_sorted_ord
    # Store all calculated values for this configuration in a dict
    # We need to use tolist() here to convert from numpy arrays to python arrays that can be serialized to json
    room_data = {
        "real_rir": real_rir.tolist(),
        "perfect_rir": perfect_rir.tolist(),
        "max_order_ism": max_order_ism,
        "geometry": {
            "room_dim": room_dim,
            "pos_src": pos_src.tolist(),
            "pos_mics": pos_mics.tolist(),
        },
        "abs_coeffs": band_abs_profiles.tolist(),
        'verite' : list_src.tolist(),#np.sort(np.linalg.norm(list_src, axis=1)),
        'ordre' : list_ordre.tolist(),
    }
    # Write the contents of this dict to a file named with the number of the room and
    # current src/rcv position iteration
    with gzip.open(output_file_path, "wt", encoding="utf-8") as f:
        json.dump(room_data, f)



def main():
    db = SOFADatabase()
    # db.list()
    # print(db['EM32_Directivity'])
    # list = download_sofa_files()

    # Reads the file containing the Eigenmike's directivity measurements
    eigenmike = MeasuredDirectivityFile("EM32_Directivity", fs=16000, interp_order=18)

    # Reads the file containing Genelec 8020 's directivity measurements
    src_dir = MeasuredDirectivityFile(
        "LS_directivity_Calibrated_GENELEC_8030B", fs=16000, interp_order=18
    )
    genelec8030 = src_dir.get_source_directivity(0, orientation=Rotation3D([0, 0], "yz", degrees=True))

    list_dir = []
    for j in range(32):
        dir_obj_Emic = eigenmike.get_mic_directivity(
            f"EM_32_{j}", orientation=Rotation3D([0, 0], "yz", degrees=True)
        )
        list_dir.append(dir_obj_Emic)

    path = db["EM32_Directivity"].path
    files = sf.open_sofa_file(path)
    pos_eigenmike = files[3]

    workdir = "./dataset_genelec_8030_near_measure_eval/"
    os.makedirs(workdir, exist_ok=True)

    # --- Configuration du logger pour écrire dans un fichier ---
    log_file = os.path.join(workdir, "stdout.txt")
    gfile_stream = open(log_file, "w", encoding="utf-8")
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    # Add another handler to also log to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Début de la génération des données.")

    configurations = []
    for room_index in range(0, num_room):
        # Dimensions of the room
        Dx, Dy, Dz = fun.generate_random_room_dimensions()
        room_dim = [Dx, Dy, Dz]
        # Generate #positions_per_room mesures in the room
        for position_index in range(positions_per_room):
            # Generate 2 random points in the room, with constraints on location
            # approx = fun.approximation_distance(0.01)
            pos_src, pos_mics = fun.generate_random_points(
                Dx, Dy, Dz, distance_src_mics , dist_walls
            )
            uncertainty_pos = (np.random.rand(3) * 2 - 1) / 10     # Random uncertainty in the position of 10 cm
            configurations.append(
                (
                    list_dir,
                    genelec8030,
                    room_dim,
                    pos_src,
                    uncertainty_pos,
                    pos_mics,
                    pos_eigenmike,
                    Path(workdir) / f"room_{room_index}_{position_index}.json.gz",
                    np.random.randint(0, 1000000)
                )
            )

    # Run the function in parallel and wait for all processes to complete
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_rirs_for_config, *config)
            for config in configurations
        ]
        # Create a progress bar that will fill up as new rirs are calculated
        with alive_bar(num_room * positions_per_room) as bar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise an exception if the future raised one
                    bar()  # pylint:disable=not-callable
                except Exception as e:
                    logger.error(f"An error occurred: {e}")

    logger.info("Génération des données terminée.")
    # Fermeture du fichier de log
    gfile_stream.close()


if __name__ == "__main__":
    main()
