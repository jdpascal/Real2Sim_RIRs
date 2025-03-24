import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path

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
num_room = 64
positions_per_room = 4
distance_src_mics = 1
dist_mur = 1
max_order_ism = 10
delay = 83
limit = 1024
crop_start = 50

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
    eigenmike: MeasuredDirectivityFile,
    src_dir: MeasuredDirectivityFile,
    room_dim: list[float],
    pos_src: np.ndarray,
    pos_mics: np.ndarray,
    pos_eigenmike,
    output_file_path: Path,
):
    """
    Calculates RIRs for the given configuration and writes result to an output JSON file.
    """
    # Orientation
    cartesian_coords = pos_mics - pos_src
    r, theta, phi = fun.cartesian_to_spherical(cartesian_coords)
    orientation = Rotation3D([theta , phi], "zy", degrees=True)
    theta_mic, phi_mic = fun.random_angles()
    orientation_mic = Rotation3D([theta_mic, phi_mic], "zy", degrees=True)
    # dir = DirectionVector(theta, phi)

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

    # Create the real room with real micro and source
    room_real = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=max_order_ism,
        materials=full_materials,
        air_absorption=True,
        ray_tracing=False,
        min_phase=True,
    )

    # Add source and microphone
    genelec8020 = src_dir.get_source_directivity(
        "Genelec_8020", orientation=orientation
    )
    room_real.add_source(pos_src, directivity=genelec8020)
    # Get the directivity objects from the files and add mics
    list_dir = []
    for j in range(32):
        dir_obj_Emic = eigenmike.get_mic_directivity(
            f"EM_32_{j}", orientation=orientation_mic
        )
        list_dir.append(dir_obj_Emic)
    room_real.add_microphone_array(
        (pos_eigenmike.T + pos_mics).T, directivity=list_dir
    )  # , directivity=list_dir

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

    # Add source and microphone omnidirectionnal
    room_perfect.add_source(pos_src)
    room_perfect.add_microphone_array((pos_eigenmike.T + pos_mics).T)

    # Compute the RIR
    room_real.compute_rir()
    room_perfect.compute_rir()
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

    real_rir = test_real[:, crop_start + delay : crop_start + limit + delay] / np.max(
        test_real[:, crop_start + delay : crop_start + limit + delay], axis=1
    ).reshape(-1, 1)
    real_rir = real_rir / np.max( np.abs(real_rir) )
    perfect_rir = test_perfect[:, crop_start : crop_start + limit] / np.max(
        test_perfect[:, crop_start : crop_start + limit], axis=1
    ).reshape(-1, 1)
    perfect_rir = perfect_rir / np.max( np.abs(perfect_rir) )
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
    }
    # Write the contents of this dict to a file named with the number of the room and
    # current src/rcv position iteration
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(room_data, f, indent=4)


def main():
    db = SOFADatabase()
    # db.list()
    # print(db['EM32_Directivity'])
    # list = download_sofa_files()

    # Reads the file containing the Eigenmike's directivity measurements
    eigenmike = MeasuredDirectivityFile("EM32_Directivity", fs=16000)

    # Reads the file containing Genelec 8020 's directivity measurements
    src_dir = MeasuredDirectivityFile(
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz", fs=16000
    )

    path = db["EM32_Directivity"].path
    files = sf.open_sofa_file(path)
    pos_eigenmike = files[3]

    workdir = "./dataset_2"
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
    for room_index in range(num_room):
        # Dimensions of the room
        Dx, Dy, Dz = fun.generate_random_room_dimensions()
        room_dim = [Dx, Dy, Dz]
        # Generate #positions_per_room mesures in the room
        for position_index in range(positions_per_room):
            # Generate 2 random points in the room, with constraints on location
            approx = fun.approximation_distance(0.1)
            pos_src, pos_mics = fun.generate_random_points(
                Dx, Dy, Dz, distance_src_mics + approx, dist_mur
            )
            configurations.append(
                (
                    eigenmike,
                    src_dir,
                    room_dim,
                    pos_src,
                    pos_mics,
                    pos_eigenmike,
                    Path(workdir) / f"room_{room_index}_{position_index}.json",
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
