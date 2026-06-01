import numpy as np

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyroomacoustics as pra
from pyroomacoustics.datasets import SOFADatabase, download_sofa_files
import pyroomacoustics.directivities.sofa as sf

import function_utils as fun2

import imageio
import os
import logging
import json
import sofa
import gzip

# -------------------------------------------------------------------------
# Global configuration
# -------------------------------------------------------------------------

# Depending on the inference_type you did
#   - 'ode'
#   - 'sde'
inference_type = 'sde'

# Matching thresholds used by the Hungarian assignment algorithm:
#   threshold_r   : radial-distance threshold
#   threshold_deg : angular threshold in degrees
threshold_r, threshold_deg = 1.0, 20.0

# Maximum distance used to separate regular image sources from "bonus" ones.
# Sources farther than this distance are treated differently in the metrics.
dist_max = 11


# -------------------------------------------------------------------------
# Microphone-array positions for each measurement
# -------------------------------------------------------------------------

# Each entry corresponds to the microphone-array position for one measurement.
# The positions are 3D coordinates: [x, y, z].
pos_mics = [
    np.array([2.63, 1.32, 1.71]),
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
    np.array([5.03, 3.24, 1.35])
]


# -------------------------------------------------------------------------
# Metric storage
# -------------------------------------------------------------------------

# Lists used to store metrics over all measurements.
RP_list, total_list, mee_list, mre_list, mae_list = [], [], [], [], []

# False-positive counters.
FP_count = 0


# -------------------------------------------------------------------------
# Main evaluation loop over measurements
# -------------------------------------------------------------------------

for measure_idx in range(1, 11):

    print(measure_idx)

    # Get microphone-array position for the current measurement.
    pos_mic = pos_mics[measure_idx - 1]

    # ---------------------------------------------------------------------
    # Load estimated image-source data
    # ---------------------------------------------------------------------

    if inference_type == 'ode':
        # File containing generated RIR / estimated image-source data
        data_file = f"./mes_{inference_type}/mes_{measure_idx % 10 + 1}_.npz"

    else:
        data_file = f"./mes_{inference_type}/mes_{measure_idx % 10 + 1}_.npz"

    with np.load(data_file) as res:
        # x: estimated source positions
        # a: associated amplitudes or scores
        # r: room impulse responses
        x, a, r, x_2, x_2_order = res['x'], res['a'], res['rir'], res['ground_truth'], res['order']

    # ---------------------------------------------------------------------
    # Filter estimated sources according to their amplitude
    # ---------------------------------------------------------------------

    # Keep only estimates whose amplitude is higher than 7 (hard coded)
    # idx_a is a Boolean mask.
    idx_a = []

    for amp in a:
        if amp > 7:
            idx_a.append(True)
        else:
            idx_a.append(False)

    x = x[idx_a]
    a = a[idx_a]


        # ---------------------------------------------------------------------
        # Specific case for the ICASSP campaign measurements: merge the two ground-truth point clouds
        # ---------------------------------------------------------------------
        
        # ground_truth_file = f"./mes/mes_{measure_idx}_th.npz"

        # with np.load(ground_truth_file) as ground_truth:
        #     # Ground-truth image-source coordinates for two configurations
        #     verite_terrain = ground_truth['verite_1']
        #     verite_terrain_2 = ground_truth['verite_2']

        #     # Corresponding reflection orders
        #     order = ground_truth['order_1']
        #     order_2 = ground_truth['order_2']

        # print("order shape:", order.shape)


        # ---------------------------------------------------------------------
        # Merge the two ground-truth point clouds while keeping reflection labels
        # ---------------------------------------------------------------------

        # if __name__ == "__main__":

        #     # Append the reflection order as a label column:
        #     # resulting shape is (N, 4): [x, y, z, order]
        #     truth_cloud_1 = np.c_[verite_terrain, order]
        #     truth_cloud_2 = np.c_[verite_terrain_2, order_2]

        #     # Merge both labeled ground-truth clouds.
        #     # Points are merged only if they have the same label and are spatially
        #     # close within tol=1e-3.
        #     merged, unmatched_Trolley, unmatched_Floor, idx_umTro_in_merged, idx_umFloor_in_merged = (
        #         fun2.merge_unique_tol_labeled(
        #             truth_cloud_2,
        #             truth_cloud_1,
        #             tol=1e-3,
        #             reduce="first"
        #         )
        #     )


        # # Ambiguities correspond to reflection on the trolley and on the floor, which is not possible to 
        # # generate a ground truth point cloud as it is only possible for shoebox rooms
        # # These are later handled specially in the metric computation.
        # ambiguities = (idx_umTro_in_merged, idx_umFloor_in_merged)

        # # Extract merged ground-truth coordinates and labels.
        # x_2 = merged[:, :3]
        # x_2_order = merged[:, 3]


    # ---------------------------------------------------------------------
    # Sort estimated sources by distance from the origin
    # ---------------------------------------------------------------------

    d_sorted = np.sort(np.linalg.norm(x, axis=1))
    idx_sorted = np.argsort(np.linalg.norm(x, axis=1))

    x_sorted = x[idx_sorted]
    a = a[idx_sorted]
    x = x_sorted

    print(
        "measurement number:",
        measure_idx,
        "a=",
        a,
        "number of estimated sources:",
        x.shape[0]
    )


    # ---------------------------------------------------------------------
    # Sort merged ground-truth sources by distance from the origin
    # ---------------------------------------------------------------------

    d_2_sorted = np.sort(np.linalg.norm(x_2, axis=1))
    idx_sorted_2 = np.argsort(np.linalg.norm(x_2, axis=1))

    x_2_sorted = x_2[idx_sorted_2]


        # ---------------------------------------------------------------------
        # Update ambiguity indices after sorting the ground-truth cloud
        # ---------------------------------------------------------------------

        # # Convert the old ambiguity indices into their new indices after sorting.
        # idx_umTro = []

        # for k in range(idx_umTro_in_merged.shape[0]):
        #     idx_umTro.append(np.where(idx_sorted_2 == idx_umTro_in_merged[k])[0])

        # idx_umTro = np.array(idx_umTro).squeeze()

        # idx_umFloor = []

        # for k in range(idx_umFloor_in_merged.shape[0]):
        #     idx_umFloor.append(np.where(idx_sorted_2 == idx_umFloor_in_merged[k])[0])

        # idx_umFloor = np.array(idx_umFloor).squeeze()

    # Apply the same sorting to the reflection orders and coordinates.
    x_2_order = x_2_order[idx_sorted_2]
    x_2 = x_2_sorted

    # Updated ambiguity tuple after sorting.
    # ambiguities = (idx_umTro.T, idx_umFloor.T)


    # ---------------------------------------------------------------------
    # Estimate the rotation aligning estimated sources with ground truth
    # ---------------------------------------------------------------------

    # Full roll search range, in degrees.
    max_angle = 360

    # Ground-truth indices located farther than dist_max.
    # These are considered "bonus" sources.
    ind_bonus = np.where(np.linalg.norm(x_2, axis=1) > dist_max)[0]

    # First, align the real source which is always the closest to the microphone array and always well estimated, i.e. the first point in both clouds.
    # we apply Rodrigues rotation to align
    R_align = fun2.rodrigues_rotation_matrix(x[0], x_2[0])

    # Apply the initial alignment.
    x_ali = x.dot(R_align.T)


    # ---------------------------------------------------------------------
    # Search for the best roll angle around the first aligned source
    # ---------------------------------------------------------------------

    best_value = None

    for angle_deg in np.arange(0, max_angle, 0.05):

        # Define roll angle phi around the axis source/microphone
        phi = np.deg2rad(angle_deg)

        # Axis of the roll: source/microphone direction
        axis_q = x_ali[0] / np.linalg.norm(x_ali[0])

        # Roll rotation matrix around axis_q.
        R_roll = fun2.rodrigues_matrix(axis_q, phi)

        # Apply roll rotation to all estimated sources.
        x_candidate = x_ali.dot(R_roll.T)

        # Total rotation from original estimated coordinates to candidate.
        R_total = R_roll.T.dot(R_align.T)


        # -----------------------------------------------------------------
        # Match candidate estimated sources to ground-truth sources
        # -----------------------------------------------------------------

        row_ind, col_ind, dists, Q_reordered, idx_bad = fun2.reorder_by_hungarian(
            x_2[:x_candidate.shape[0]],
            x_candidate,
            threshold_r,
            threshold_deg
        )

        # Remove bad matches, i.e. pairs that exceed the matching thresholds.
        row_ind = np.array([i for i in row_ind if i not in row_ind[idx_bad]])
        col_ind = np.array([i for i in col_ind if i not in col_ind[idx_bad]])

        # Skip this rotation if fewer than five valid matches were found.
        if col_ind.shape[0] < 5:  # 5 is here hard coded because we want to align at least the first 5, then the others should be aligned as well if the roll is correct, otherwise the angular error will be high.
            continue

        # Normalize matched points to compare angular alignment.
        points_rotated_norm = (
            x_candidate[col_ind] /
            np.linalg.norm(x_candidate[col_ind], axis=1, keepdims=True)
        )

        x_2_norm = (
            x_2[row_ind] /
            np.linalg.norm(x_2[row_ind], axis=1, keepdims=True)
        )

        # Compute the mean angular error on matches 1 to 4.
        # The first match is skipped, likely because it was already used
        # to define the initial alignment axis.
        value = np.mean(
            np.rad2deg(
                np.diag(
                    np.arccos(
                        np.dot(points_rotated_norm[1:5, :], x_2_norm[1:5, :].T)
                    )
                )
            )
        )

        # Keep the rotation that minimizes the angular error.
        if best_value is None or value < best_value:
            best_value = value
            best_rotation = R_total
            best_roll = R_roll.T
            best_phi = phi

    print("best phi (deg):", np.rad2deg(best_phi))


    # ---------------------------------------------------------------------
    # Apply the best roll rotation
    # ---------------------------------------------------------------------

    points_rotated = x_ali.dot(best_roll)


    # ---------------------------------------------------------------------
    # Final matching between rotated estimates and ground truth
    # ---------------------------------------------------------------------

    row_ind, col_ind, dists, Q_reordered, idx_bad = fun2.reorder_by_hungarian(
        points_rotated,
        x_2,
        threshold_r,
        threshold_deg
    )

    # Remove bad matches.
    row_ind = np.array([i for i in row_ind if i not in row_ind[idx_bad]])
    col_ind = np.array([i for i in col_ind if i not in col_ind[idx_bad]])


    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------

    # Room geometry translated relative to the microphone-array position.
    points = fun2.points_from_idx_measure(measure_idx)

    show_visualization = True

    if show_visualization:
        # Matched estimated points.
        P = points_rotated[row_ind]

        # Matched ground-truth points.
        Q = x_2[col_ind]

        # All the ground-truth points before the bonus-distance threshold.
        B = x_2[:ind_bonus[0]]

        # All rotated estimated points.
        BB = points_rotated

        # Display an interactive Plotly visualization:
        #   - matched estimated sources
        #   - matched ground-truth sources
        #   - expected ground-truth sources
        #   - all estimated sources
        #   - room outline
        fun2.show_point_cloud_plotly(
            P,
            Q,
            B,
            BB,
            index=(np.arange(0, row_ind.shape[0]), np.arange(0, row_ind.shape[0])),
            labels=["Estimated", "Ground truth", "Ground truth", "Estimated"],
            room=points,
            title=f"Visualization of image-source estimation from measurement {measure_idx}",
            save_html=f"point_clouds_plotly_{measure_idx}.html",
            camera=np.array([1.5, -2.15, 0.77]) * 0.8
        )


    # ---------------------------------------------------------------------
    # Metric computation
    # ---------------------------------------------------------------------
    # ambiguities = (idx_umTro.T, idx_umFloor.T) #if there are ambiguities, otherwise set to (None, None)
    ambiguities = (None, None) #if there are no ambiguities

    RP, total, mee, mre, mae, FP = fun2.metrics(
        points_rotated,
        x_2,
        ambiguities,
        row_ind,
        col_ind,
        idx_bad,
        x_2_order,
        ind_bonus,
        threshold_r,
        threshold_deg,
        max_dist=dist_max
    )

    # Accumulate false positives over all measurements.
    FP_count += FP

    # Store per-measurement metrics.
    RP_list.append(RP)
    total_list.append(total)
    mee_list.append(mee)
    mre_list.append(mre)
    mae_list.append(mae)