import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def compare_arrays(a, b):
    """Compute the distances between each line vector of two 2d-arrays a and b and match the smallest distances from
    the lines of a to the lines of b (compare_arrats(a, b) != compare_arrays(b, a)).

    Return : tuple (ind, dist) where :
        -ind is a flat array such as ind[i] is the line of b closest to the ith line of a
        -dist is a flat array containing the corresponding distances """

    # a,b have shape (N1, d), (N2,d), dist has shape N1, N2
    dist = np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :])**2, axis=-1))
    # shape N1
    ind_min = np.argmin(dist, axis=1)
    return ind_min, dist[np.arange(a.shape[0]), ind_min]

def unique_matches(a, b, ampl=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the distances between each line vector of two 2d-arrays a and b and match the smallest distances from
    the lines of a to the lines of b. If more than one line of a is matched to the same line of b, the array ampl can be
    used to eliminate the duplicate matches. If ampl is None, the smallest distance is retained.
    
    Args:-ampl (ndarray):flat array of amplitudes used to sort the matches. If a[k] and a[l] are matched to b[m] and
    ampl[k] < ampl[l] only a[l] is considered. If all amplitudes are equal the smallest index wins. If ampl is None,
    return the closest match in distance.
    
    Return: tuple(inda, indb, dist) where :
        -inda, indb are the arrays giving the index of matches between the lines of a and b.
    len(inda) = len(indb) <= len(b) (at most len(b) unique matches can be found).
        -dist is the corresponding array of distances
    """

    ind_min, dist = compare_arrays(a,b)  # for each line of a, compute the closest line of b
    unique = np.unique(ind_min)  # get the indices of the lines of b closest to those of a, without repetitions
    final_inda_list, final_indb_list, final_dist_list, idx_bad = [], [], [], []
    for ind in unique:  # loop over the indices of the lines of b closest to a
        matches = ind_min == ind  # indices of the lines of a matched to the line ind of b
        if ampl is None:
            tmp_dist = np.full_like(dist, np.inf, dtype='float')
            tmp_dist[matches] = dist[matches]
            best_match = np.argmin(tmp_dist)  # index of the line of a closest to the line indexed by ind in b
        else:
            tmp_ampl = np.zeros_like(ampl)
            tmp_ampl[matches] = ampl[matches]
            best_match = np.argmax(tmp_ampl)
        if np.abs(np.linalg.norm(a[best_match]) - np.linalg.norm(b[ind])) < 1.0 and np.rad2deg(np.arccos((a[best_match] / np.linalg.norm(a[best_match]))@ (b[ind] / np.linalg.norm(b[ind])).T)) < 21.0:
            final_inda_list.append(best_match)
            final_indb_list.append(ind)
            final_dist_list.append(dist[best_match])
        else : 
            idx_bad.append(best_match)

    return np.array(final_inda_list), np.array(final_indb_list), np.array(final_dist_list), np.array(idx_bad)
def plane_perp_to_vector(v, p0=None, normalize=False):
    """
    Plane perpendicular to vector v in 3D, passing through p0
    (the origin by default).

    Returns:
      - (A, B, C, d) such that: A x + B y + C z = d
      - normal vector n, optionally normalized
      - two unit vectors (u, w) spanning the plane
        as an orthonormal basis
    """
    v = np.asarray(v, dtype=float).reshape(3)
    if np.allclose(v, 0):
        raise ValueError("Zero vector: the normal plane is not defined.")
    p0 = np.zeros(3) if p0 is None else np.asarray(p0, dtype=float).reshape(3)

    n = v / np.linalg.norm(v) if normalize else v.copy()

    # Cartesian equation of the plane:
    # n·(x - p0) = 0  <=>  n·x = n·p0
    A, B, C = n
    d = float(n @ p0)

    # Build an orthonormal basis (u, w) of the plane
    # Choose the axis least aligned with n
    axes = np.eye(3)
    a = axes[np.argmin(np.abs(n))]  # (1,0,0), (0,1,0), or (0,0,1)

    # Project a onto the plane and normalize
    u = a - (a @ n) * n / (n @ n)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        # Extremely aligned case: choose another axis
        a = axes[(np.argmin(np.abs(n)) + 1) % 3]
        u = a - (a @ n) * n / (n @ n)
        u_norm = np.linalg.norm(u)

    u /= u_norm
    w = np.cross(n, u)
    w /= np.linalg.norm(w)

    return (A, B, C, d), n, (u, w)


def signed_distance_point_to_plane(x, plane):
    """
    x: point(s), shape (3,) or (N, 3)
    plane: (A, B, C, d) such that A x + B y + C z = d

    Returns the signed distance, positive if x lies on the side
    pointed to by the normal vector (A, B, C).
    """
    A, B, C, d = plane
    n = np.array([A, B, C], dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-15:
        raise ValueError("Zero normal vector for the plane.")

    X = np.atleast_2d(np.asarray(x, dtype=float))
    sdist = (X @ n - d) / n_norm
    return sdist if X.shape[0] > 1 else float(sdist[0])


def rodrigues_matrix(axis, angle):
    """
    Rotation matrix around `axis` (3-vector) by an angle `angle` in radians.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([[     0, -axis[2],  axis[1]],
                  [ axis[2],      0, -axis[0]],
                  [-axis[1],  axis[0],      0]])
    c, s = np.cos(angle), np.sin(angle)
    return np.eye(3)*c + (1-c)*np.outer(axis, axis) + s*K


def rodrigues_rotation_matrix(p, q, eps=1e-8):
    """
    Computes the Rodrigues rotation matrix that rotates p toward q.

    p, q: 3D vectors, either shape (1, 3) or shape (3,)
    """
    # Unit vectors
    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)
    u = p / norm_p
    v = q / norm_q

    # Rotation axis
    k = np.cross(u, v)
    if np.linalg.norm(k) < eps:
        # Collinear or anti-collinear case: identity matrix
        return np.eye(3)

    k = k / np.linalg.norm(k)

    # Angle
    cosθ = np.clip(np.dot(u, v), -1.0, 1.0)
    θ = np.arccos(cosθ)

    # Skew-symmetric matrix of k
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])

    # Rodrigues formula
    R = np.eye(3) * np.cos(θ) \
        + (1 - np.cos(θ)) * np.outer(k, k) \
        + np.sin(θ) * K

    return R


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables proj='3d')


def show_point_cloud_matplotlib(*clouds, colors=None, labels=None, size=10, alpha=0.9,
                                title="3D point cloud", elev=20, azim=35):
    """
    clouds: one or more arrays with shape (N_i, 3)
    colors: list of Matplotlib colors, e.g. 'tab:blue', '#ff0000', (r, g, b)
    labels: legend labels, same length as clouds
    size: marker size
    alpha: point transparency
    elev, azim: initial viewing angles
    """
    if not clouds:
        raise ValueError("Provide at least one point cloud array with shape Nx3.")

    clouds = [np.asarray(c, float).reshape(-1, 3) for c in clouds]
    k = clouds[0].shape[0] if clouds else 0

    if colors is None:
        palette = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange',
                   'tab:purple', 'tab:brown', 'tab:cyan']
        colors = [palette[i % len(palette)] for i in range(k)]

    if labels is None:
        labels = [f"Cloud {i+1}" for i in range(k)]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for each point cloud
    for pts, col, lab in zip(clouds, colors, labels):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=size, c=col, alpha=alpha, depthshade=True, label=lab)

    # Axes and view
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left")

    # Isotropic scaling for correct proportions
    all_pts = np.vstack(clouds)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    max_range = (maxs - mins).max()

    rx = ry = rz = max_range / 2.0 if max_range > 0 else 1.0

    ax.set_xlim(centers[0] - rx, centers[0] + rx)
    ax.set_ylim(centers[1] - ry, centers[1] + ry)
    ax.set_zlim(centers[2] - rz, centers[2] + rz)

    try:
        # Matplotlib >= 3.3
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


# ----------------- Generic tools -----------------

def concat_clouds(P, Q):
    """Concatenates two point clouds with shapes (N, 3) and (M, 3)."""
    P = np.asarray(P, float).reshape(-1, 3)
    Q = np.asarray(Q, float).reshape(-1, 3)
    return np.vstack([P, Q])


def merge_unique_tol(P, Q, tol=1e-6, reduce="first"):
    """
    Merges P and Q while removing duplicates according to a spatial tolerance `tol`.

    Method:
        Points are quantized on a grid with step size `tol`.

    reduce:
        'first' -> keeps one representative point
        'mean'  -> returns the centroid of each cell
    """
    PQ = concat_clouds(P, Q)
    keys = np.round(PQ / tol).astype(np.int64)  # cell key
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    if reduce == "first":
        # Retrieve the first point of each cell
        order = np.argsort(inv, kind="mergesort")
        first_pos = order[np.unique(inv[order], return_index=True)[1]]
        return PQ[first_pos]

    elif reduce == "mean":
        sums = np.zeros((len(uniq), 3), float)
        np.add.at(sums, inv, PQ)
        counts = np.bincount(inv)
        return sums / counts[:, None]

    else:
        raise ValueError("reduce must be either 'first' or 'mean'.")


# ----------------- Generic tools -----------------

def concat_clouds(P, Q):
    """Concatenates two point clouds with shapes (N, 3) and (M, 3)."""
    P = np.asarray(P, float).reshape(-1, 3)
    Q = np.asarray(Q, float).reshape(-1, 3)
    return np.vstack([P, Q])


def merge_unique_tol(P, Q, tol=1e-6, reduce="first"):
    """
    Merges P and Q while removing duplicates according to a spatial tolerance `tol`.

    Method:
        Points are quantized on a grid with step size `tol`.

    reduce:
        'first' -> keeps one representative point
        'mean'  -> returns the centroid of each cell
    """
    PQ = concat_clouds(P, Q)
    keys = np.round(PQ / tol).astype(np.int64)  # cell key
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    if reduce == "first":
        # Retrieve the first point of each cell
        order = np.argsort(inv, kind="mergesort")
        first_pos = order[np.unique(inv[order], return_index=True)[1]]
        return PQ[first_pos]

    elif reduce == "mean":
        sums = np.zeros((len(uniq), 3), float)
        np.add.at(sums, inv, PQ)
        counts = np.bincount(inv)
        return sums / counts[:, None]

    else:
        raise ValueError("reduce must be either 'first' or 'mean'.")


def merge_unique_tol_labeled(P, Q, tol=1e-6, reduce="first"):
    """
    Merges two labeled point clouds (x, y, z, label) while removing duplicates
    within tolerance `tol`.

    A merge occurs only if both the label and the quantized spatial cell match.

    reduce:
        'first' -> keeps one representative point from the cluster
        'mean'  -> returns the centroid of the cluster

    Returns
    -------
    merged: (K, 4)
        Merged point cloud, one point per cluster.

    unmatched_P: (Kp, 4)
        Points from P whose cluster does not exist in Q.

    unmatched_Q: (Kq, 4)
        Points from Q whose cluster does not exist in P.

    idx_umP_in_merged: (Kp,)
        Indices in `merged` corresponding to each row of `unmatched_P`.
        Several rows may point to the same index.

    idx_umQ_in_merged: (Kq,)
        Same as above for `unmatched_Q`.
    """
    P = np.asarray(P)
    Q = np.asarray(Q)

    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 4 or Q.shape[1] != 4:
        raise ValueError("P and Q must have shapes (N, 4) and (M, 4): [x, y, z, label].")

    if tol <= 0:
        raise ValueError("tol must be > 0.")

    # Separate coordinates and labels
    P_xyz, P_lab = P[:, :3].astype(float), P[:, 3].astype(np.int64)
    Q_xyz, Q_lab = Q[:, :3].astype(float), Q[:, 3].astype(np.int64)

    # Spatial quantization
    P_cell = np.round(P_xyz / tol).astype(np.int64)
    Q_cell = np.round(Q_xyz / tol).astype(np.int64)

    # Cluster keys = (label, cell_x, cell_y, cell_z)
    keysP = np.c_[P_lab, P_cell]
    keysQ = np.c_[Q_lab, Q_cell]

    # Concatenate to group clusters globally
    PQ_xyz = np.vstack([P_xyz, Q_xyz])
    PQ_lab = np.concatenate([P_lab, Q_lab])
    keysPQ = np.vstack([keysP, keysQ])

    uniq, inv = np.unique(keysPQ, axis=0, return_inverse=True)  # K clusters
    K = len(uniq)

    invP = inv[:len(P)]
    invQ = inv[len(P):]

    cntP = np.bincount(invP, minlength=K)
    cntQ = np.bincount(invQ, minlength=K)

    onlyP = (cntP > 0) & (cntQ == 0)
    onlyQ = (cntQ > 0) & (cntP == 0)

    # Build merged cloud + mapping from cluster ID to merged index
    if reduce == "first":
        order = np.argsort(inv, kind="mergesort")
        first_pos = order[np.unique(inv[order], return_index=True)[1]]
        merged_xyz = PQ_xyz[first_pos]
        merged_lab = PQ_lab[first_pos]
        merged = np.c_[merged_xyz, merged_lab]

        # Mapping: cluster_id -> row index in merged
        clusters_order = inv[first_pos]
        cluster_to_merged = np.empty(K, dtype=np.int64)
        cluster_to_merged[clusters_order] = np.arange(K)

    elif reduce == "mean":
        sums = np.zeros((K, 3), float)
        np.add.at(sums, inv, PQ_xyz)
        counts = np.bincount(inv, minlength=K).astype(float)

        merged_xyz = sums / counts[:, None]
        merged_lab = uniq[:, 0].astype(np.int64)
        merged = np.c_[merged_xyz, merged_lab]

        # In this case, the merged order is the same as the cluster order
        cluster_to_merged = np.arange(K, dtype=np.int64)

    else:
        raise ValueError("reduce must be either 'first' or 'mean'.")

    # Unmatched points and corresponding indices in merged
    mask_umP = onlyP[invP]
    mask_umQ = onlyQ[invQ]

    unmatched_P = P[mask_umP]
    unmatched_Q = Q[mask_umQ]

    idx_umP_in_merged = cluster_to_merged[invP[mask_umP]]
    idx_umQ_in_merged = cluster_to_merged[invQ[mask_umQ]]

    return merged, unmatched_P, unmatched_Q, idx_umP_in_merged, idx_umQ_in_merged


def plane_from_3pts(p1, p2, p3, normalize=True, return_basis=False):
    """
    Computes the plane passing through three 3D points p1, p2, and p3.

    Parameters
    ----------
    p1, p2, p3: array-like, shape (3,)
        Three non-collinear points.

    normalize: bool
        If True, returns a unit normal vector (A, B, C).
        The equation remains A x + B y + C z = d, with d scaled accordingly.

    return_basis: bool
        If True, also returns two unit vectors (u, w) forming a basis of the plane.

    Returns
    -------
    (A, B, C, d):
        Cartesian equation of the plane:
        A x + B y + C z = d,
        where (A, B, C) is the normal vector.

    n:
        The normal vector used, unit-length if normalize=True.

    (u, w): optional
        Two orthogonal unit vectors spanning the plane, if return_basis=True.

    Notes
    -----
    - The orientation of the normal follows the right-hand rule on
      (p2 - p1) × (p3 - p1).
    - Raises ValueError if the points are collinear or too close to each other.
    """
    p1 = np.asarray(p1, dtype=float).reshape(3)
    p2 = np.asarray(p2, dtype=float).reshape(3)
    p3 = np.asarray(p3, dtype=float).reshape(3)

    v1 = p2 - p1
    v2 = p3 - p1

    n = np.cross(v1, v2)  # raw normal vector
    n_norm = np.linalg.norm(n)

    if n_norm < 1e-12 * (np.linalg.norm(v1) + np.linalg.norm(v2)):
        raise ValueError("Collinear or nearly collinear points: the plane is not defined.")

    if normalize:
        n = n / n_norm  # unit normal vector

    A, B, C = n
    d = float(n @ p1)  # because n · x = n · p1 => A x + B y + C z = d

    if not return_basis:
        return (A, B, C, d), n

    # Orthonormal basis of the plane (u, w)
    # Choose the axis least aligned with n to build u in the plane
    axes = np.eye(3)
    a = axes[np.argmin(np.abs(n))]

    # Project a onto the plane perpendicular to n
    u = a - (a @ n) * n / (n @ n)

    if np.linalg.norm(u) < 1e-12:
        a = axes[(np.argmin(np.abs(n)) + 1) % 3]
        u = a - (a @ n) * n / (n @ n)

    u = u / np.linalg.norm(u)
    w = np.cross(n, u)
    w = w / np.linalg.norm(w)

    return (A, B, C, d), n, (u, w)


def signed_distance_point_to_plane(x, plane):
    """
    x: point(s), shape (3,) or (N, 3)
    plane: (A, B, C, d) such that A x + B y + C z = d

    Returns the signed distance, positive if x lies on the side
    pointed to by the normal vector (A, B, C).
    """
    A, B, C, d = plane
    n = np.array([A, B, C], dtype=float)
    n_norm = np.linalg.norm(n)

    if n_norm < 1e-15:
        raise ValueError("Zero normal vector for the plane.")

    X = np.atleast_2d(np.asarray(x, dtype=float))
    sdist = (X @ n - d) / n_norm

    return sdist if X.shape[0] > 1 else float(sdist[0])


import plotly.graph_objects as go


def show_point_cloud_plotly(*clouds, index=None, colors=None, labels=None, size=None,
                            save_html=None, title="3D point cloud", markers=None,
                            room=None, camera=[1.5, -1.5, 1.5]):
    """
    clouds: one or more arrays with shape (N_i, 3)
    index: indices of matched pairs between the first and second point clouds
    colors: list of Plotly colors, e.g. 'royalblue', 'crimson', 'green', or None
    labels: legend names for each point cloud
    size: point size
    save_html: path to a .html file to save an interactive version, optional
    markers: list of Plotly markers, e.g. 'x', 'circle-open', or None
    """
    if not clouds:
        raise ValueError("Provide at least one point cloud array with shape Nx3.")

    k = len(clouds)

    if colors is None:
        palette = ["royalblue", "crimson", "crimson", "royalblue",
                   "seagreen", "seagreen", "purple", "goldenrod", "teal"]
        colors = [palette[i % len(palette)] for i in range(k)]

    if labels is None:
        labels = [f"Cloud {i+1}" for i in range(k)]

    if markers is None:
        palette = ["x", "circle-open", "circle-open", "x"]
        markers = [palette[i % len(palette)] for i in range(k)]

    if size is None:
        palette = [3.5, 9, 9, 3.5]
        size = [palette[i % len(palette)] for i in range(k)]

    fig = go.Figure()

    for pts, col, lab, mark, size_ in zip(clouds, colors, labels, markers, size):
        pts = np.asarray(pts, float).reshape(-1, 3)

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(symbol=mark, size=size_, opacity=0.9, color=col),
            name=lab or None
        ))

    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        marker=dict(symbol="diamond", size=5, opacity=0.9, color='rgb(255,0,0)'),
        name="Mic. array"
    ))

    # Add lines between matched points
    if k >= 2 and index is not None:
        row_ind, col_ind = index

        P, Q = clouds[0], clouds[1]
        P = np.asarray(P, float).reshape(-1, 3)
        Q = np.asarray(Q, float).reshape(-1, 3)

        pairs = list(zip(row_ind, col_ind))
        width = 4

        X, Y, Z = [], [], []

        for i, j in pairs:
            X += [P[i, 0], Q[j, 0], None]
            Y += [P[i, 1], Q[j, 1], None]
            Z += [P[i, 2], Q[j, 2], None]

        fig.add_trace(go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='lines',
            line=dict(width=width, color='rgb(0,120,0)')
        ))

        fig.update_layout(
            scene=dict(aspectmode='data'),
            title=title or "Pairs between two point clouds"
        )

    if room is not None:
        points = room
        pairs = list(zip(row_ind, col_ind))
        width = 1

        X, Y, Z = [], [], []

        for i in range(points.shape[0] - 1):
            X += [points[i, 0], points[i + 1, 0], None]
            Y += [points[i, 1], points[i + 1, 1], None]
            Z += [points[i, 2], points[i + 1, 2], None]

        fig.add_trace(go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='lines',
            line=dict(width=width, color='rgb(0,0,0)')
        ))

        fig.update_layout(
            scene=dict(aspectmode='data'),
            title=title or "Pairs between two point clouds"
        )

    fig.update_layout(
        title=title,
        title_font_size=20,
        width=1100,
        height=1000,
        scene=dict(
            xaxis=dict(
                title=dict(text="X (m)", font=dict(size=24)),
            ),
            yaxis=dict(
                title=dict(text="Y (m)", font=dict(size=24)),
            ),
            zaxis=dict(
                title=dict(text="Z (m)", font=dict(size=24)),
            ),
            aspectmode="data"
        ),
        legend=dict(itemsizing="constant")
    )

    fig.update_scenes(
        camera_eye_x=camera[0],
        camera_eye_y=camera[1],
        camera_eye_z=camera[2],
        camera_center_x=0,
        camera_center_y=0,
        camera_center_z=0,
        xaxis_title_font=dict(size=24),
        yaxis_title_font=dict(size=24),
        zaxis_title_font=dict(size=24),
        xaxis_tickfont=dict(size=16),
        yaxis_tickfont=dict(size=16),
        zaxis_tickfont=dict(size=16),
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn", auto_open=True)
        print(f"Interactive file saved: {save_html}")
        fig.write_image("point_clouds.svg")

    fig.show()


def points_from_idx_measure(idx):
    mesure = {
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
                    np.array([2.94, 3.45, 1.71]),
                    np.array([1.96, 3.08, 1.71]),
                    np.array([3.11, 3.29, 1.71]),
                    np.array([3.76, 3.3, 1.71]),
                    np.array([7.32, 2.87, 1.35]),
                    np.array([7.51, 2.06, 1.35]),
                    np.array([9.53 - 1.69, 1.71, 1.35]),
                    np.array([9.53 - 3.27, 2.14, 1.35]),
                    np.array([9.53 - 3.27, 2.14, 1.35]),
                    np.array([5.00, 1.58, 1.35])],

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

    room_dim = mesure["room_dim"][idx - 1]
    Lx, Ly, Lz = room_dim[0], room_dim[1], room_dim[2]

    # Vertices: 0/1 = min/max along each axis
    O = np.array([0.0, 0.0, 0.0])        # (0, 0, 0)
    X = np.array([Lx, 0.0, 0.0])         # (Lx, 0, 0)
    XY = np.array([Lx, Ly, 0.0])         # (Lx, Ly, 0)
    Y = np.array([0.0, Ly, 0.0])         # (0, Ly, 0)
    OZ = np.array([0.0, 0.0, Lz])        # (0, 0, Lz)
    XZ = np.array([Lx, 0.0, Lz])         # (Lx, 0, Lz)
    XYZ = np.array([Lx, Ly, Lz])         # (Lx, Ly, Lz)
    YZ = np.array([0.0, Ly, Lz])         # (0, Ly, Lz)

    # Polyline path: each consecutive pair is an edge of the cuboid
    path = np.vstack([
        O, X, XY, Y, O,              # bottom loop
        OZ, XZ, XYZ, YZ, OZ,         # top loop
        XZ, X, XY, XYZ, YZ, Y, O     # vertical edges and returns
    ])

    return path - mesure["pos_mics"][idx - 1]


def merge_close_points(points, threshold=1.0):
    """
    Merges points with shape (N, 3) whose distance is < threshold.

    The merge is performed by replacing close points with their mean.

    Args:
        points (ndarray): shape (N, 3)
        threshold (float): maximum distance for merging

    Returns:
        ndarray: new merged points
    """
    pts = points.copy()
    N = len(pts)
    merged = np.zeros(N, dtype=bool)
    clusters = []

    for i in range(N):
        if merged[i]:
            continue

        # Compute distances from point i to all other points
        dists = np.linalg.norm(pts - pts[i], axis=1)
        mask = dists < threshold
        merged |= mask
        clusters.append(pts[mask].mean(axis=0))

    return np.vstack(clusters)


def metrics_simul(estimation, cible, row_ind, col_ind, idx_bad, ordre,
                  ind_bonus, threshold_r, threshold_deg):
    RP = []
    total = []
    precision = []
    recall = []
    mee_list = []
    mre_list = []
    mae_list = []
    FP = 0

    for i in range(1, 4):
        ind = (np.argwhere(ordre == i).T)[0]  # indices in the ground truth at order i
        not_bonus = np.sum(np.linalg.norm(cible[ind], axis=1) < 6.32)

        # indices in the ground truth at order i that were correctly estimated
        ind_i = np.array([k for k in range(col_ind.shape[0]) if col_ind[k] in ind])

        if ind_i.shape[0] == 0:
            mee_list.append(np.array([]))
            mre_list.append(np.array([]))
            mae_list.append(np.array([]))
            RP.append(0)
            total.append(not_bonus)
            continue

        col_ind_i = col_ind[ind_i]  # matched indices in the ground truth
        row_ind_i = row_ind[ind_i]

        bonus = np.sum(np.linalg.norm(cible[col_ind_i], axis=1) > 6.32)
        print(bonus)

        mee = np.linalg.norm(estimation[row_ind_i] - cible[col_ind_i], axis=1)
        mre = np.abs(
            np.linalg.norm(estimation[row_ind_i], axis=1)
            - np.linalg.norm(cible[col_ind_i], axis=1)
        )

        scalar = (
            estimation[row_ind_i] / np.linalg.norm(estimation[row_ind_i], axis=1, keepdims=True)
        ) @ (
            cible[col_ind_i] / np.linalg.norm(cible[col_ind_i], axis=1, keepdims=True)
        ).T

        mae = np.rad2deg(np.diag(np.arccos(scalar)))

        RP.append(row_ind_i.shape[0])  # real positive estimations
        total.append(bonus + not_bonus)
        mee_list.append(np.array(mee))
        mre_list.append(np.array(mre))
        mae_list.append(np.array(mae))

    FP = estimation.shape[0] - np.sum(np.array(RP))

    return RP, total, mee_list, mre_list, mae_list, FP


def metrics(estimation, cible, ambiguitie, row_ind, col_ind, idx_bad, ordre,
            ind_bonus, threshold_r, threshold_deg, max_dist=6.32):
    """
    Computes matching metrics per order between estimation and target vectors,
    taking Chariot/Floor ambiguities into account.

    Robust version:
      - avoids division by zero,
      - clamps cosine values to [-1, 1] before arccos,
      - uses safe indexing with np.flatnonzero and np.isin,
      - guarantees total >= 0,
      - fixes the Chariot/Floor ambiguity loop using the shared minimum size.

    Args:
        estimation: (N_est, D) estimated vectors
        cible: (N_gt, D) ground-truth vectors
        ambiguitie: tuple (ind_Cha, ind_Sol), ambiguous GT indices
        row_ind: estimation-side indices for matches from the Hungarian algorithm
        col_ind: target-side indices for matches from the Hungarian algorithm
        idx_bad: unused here, kept for compatibility
        ordre: (N_gt,) order value in {1, 2, 3, ...} for each target
        ind_bonus: threshold index; ind_bonus[0] is used as the upper bound
        threshold_r, threshold_deg: thresholds passed to reorder_by_hungarian_2

    Returns:
        RP: [3] true positives per order
        total: [3] expected targets per order, adjusted with bonus/not_bonus, >= 0
        mee_list: [3] arrays of Euclidean errors per order
        mre_list: [3] arrays of norm errors per order
        mae_list: [3] arrays of angular errors in degrees per order
        FP: int, global false positives
    """
    NORM_BONUS_THRESH = max_dist  # threshold kept for compatibility

    RP, total = [], []
    mee_list, mre_list, mae_list = [], [], []

    # Make ind_bonus safe and extract the limit_bonus bound
    if ind_bonus is None:
        ind_bonus_arr = np.array([], dtype=int)
    else:
        ind_bonus_arr = np.atleast_1d(ind_bonus)

    limit_bonus = int(ind_bonus_arr[0]) if ind_bonus_arr.size > 0 else 0

    # Resolve Chariot/Floor ambiguity
    ind_Cha, ind_Sol = ambiguitie

    if getattr(ind_Cha, "size", 0) == 0 or getattr(ind_Sol, "size", 0) == 0:
        ind_Cha_i = np.array([], dtype=int)
        ind_Sol_i = np.array([], dtype=int)
    else:
        ind_Cha_i, ind_Sol_i, _, _, _ = reorder_by_hungarian(
            cible[ind_Cha], cible[ind_Sol], threshold_r, threshold_deg
        )

    ind_Cha = ind_Cha[ind_Cha_i] if ind_Cha_i.size > 0 else np.array([], dtype=int)
    ind_Sol = ind_Sol[ind_Sol_i] if ind_Sol_i.size > 0 else np.array([], dtype=int)

    def safe_norm(x, axis=1):
        return np.linalg.norm(x, axis=axis)

    # Loop over orders
    for i in (1, 2, 3, 4, 5, 6, 7):
        ind = np.flatnonzero(ordre == i)

        if getattr(col_ind, "size", 0) > 0:
            ind_i = np.flatnonzero(np.isin(col_ind, ind))
        else:
            ind_i = np.array([], dtype=int)

        col_ind_i = col_ind[ind_i] if ind_i.size > 0 else np.array([], dtype=int)
        row_ind_i = row_ind[ind_i] if ind_i.size > 0 else np.array([], dtype=int)

        # bonus / not_bonus
        bonus = (
            int(np.sum(safe_norm(cible[col_ind_i], axis=1) > NORM_BONUS_THRESH))
            if col_ind_i.size > 0 else 0
        )

        not_bonus = (
            int(np.sum(safe_norm(cible[ind], axis=1) < NORM_BONUS_THRESH))
            if ind.size > 0 else 0
        )

        # Ambiguity adjustment:
        # only < limit_bonus and same order
        if ind_Cha.size > 0 and ind_Sol.size > 0 and limit_bonus > 0:
            cha_ord = np.array(
                [idx for idx in ind_Cha if idx < limit_bonus and ordre[idx] == i],
                dtype=int
            )

            sol_ord = np.array(
                [idx for idx in ind_Sol if idx < limit_bonus and ordre[idx] == i],
                dtype=int
            )

            m = int(min(cha_ord.size, sol_ord.size))

            if m > 0:
                matched_set = set(col_ind_i.tolist()) if col_ind_i.size > 0 else set()

                for a in range(m):
                    c_cha = cha_ord[a] in matched_set
                    c_sol = sol_ord[a] in matched_set

                    if (c_cha ^ c_sol) or (not c_cha and not c_sol):
                        not_bonus -= 1

        # Errors on matched pairs
        if row_ind_i.size > 0:
            diff = estimation[row_ind_i] - cible[col_ind_i]
            mee = safe_norm(diff, axis=1)

            est_n = safe_norm(estimation[row_ind_i], axis=1)
            tgt_n = safe_norm(cible[col_ind_i], axis=1)

            mre = np.abs(est_n - tgt_n)

            denom = est_n * tgt_n

            with np.errstate(divide='ignore', invalid='ignore'):
                cos = np.sum(estimation[row_ind_i] * cible[col_ind_i], axis=1) / denom

            cos = np.clip(cos, -1.0, 1.0)
            cos[denom == 0] = np.nan

            mae = np.rad2deg(np.arccos(cos))

            RP.append(int(row_ind_i.shape[0]))
            mee_list.append(mee)
            mre_list.append(mre)
            mae_list.append(mae)

        else:
            RP.append(0)
            mee_list.append(np.array([], dtype=float))
            mre_list.append(np.array([], dtype=float))
            mae_list.append(np.array([], dtype=float))

        total.append(max(int(bonus + not_bonus), 0))

    FP = max(int(estimation.shape[0] - np.sum(RP)), 0)

    return RP, total, mee_list, mre_list, mae_list, FP


from scipy.optimize import linear_sum_assignment


def reorder_by_hungarian(
    P, Q, threshold_r, threshold_deg,
    lambda_unmatch=1.0,    # penalty for leaving an element unmatched, in normalized cost units
    cap_clip=5.0,          # cost clipping cap; should generally be >= lambda_unmatch
    big_cost=1e9           # huge cost for numerically invalid pairs
):
    """
    Matches P (N, d) to Q (M, d), allowing unmatched elements through dummy rows/columns.

    Pairwise cost:
        cost(p, q) = max(
            |‖p‖ - ‖q‖| / threshold_r,
            angle(p, q) / threshold_deg
        )

    Args:
        threshold_r: radial threshold, e.g. in meters
        threshold_deg: angular threshold, in degrees
        lambda_unmatch: unmatched penalty in normalized units.
                        Typically 1.0 means a match is accepted only if c <= 1,
                        i.e. below the thresholds.
        cap_clip: clipping cap for costs; should remain >= lambda_unmatch.
        big_cost: huge cost for numerically undefined pairs.

    Returns:
        row_ind:
            indices in P of matched real elements, excluding dummies

        col_ind:
            corresponding real indices in Q

        dists:
            Euclidean distances ‖p - q‖ for the kept pairs

        Q_reordered:
            Q reordered according to P; NaN for unmatched P elements

        idx_bad:
            indices in row_ind/col_ind of pairs exceeding the thresholds, c > 1
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    N, D = P.shape[0], P.shape[1]
    M = Q.shape[0]

    # Utilities
    def angles_deg(A, B):
        rA = np.linalg.norm(A, axis=1)
        rB = np.linalg.norm(B, axis=1)

        dots = A @ B.T
        denom = np.outer(rA, rB)

        with np.errstate(divide='ignore', invalid='ignore'):
            cos = dots / denom

        cos = np.clip(cos, -1.0, 1.0)
        cos[~np.isfinite(cos)] = np.nan  # zero norms -> NaN

        ang = np.rad2deg(np.arccos(cos))

        return ang, rA, rB

    # Normalized costs over the whole matrix
    Ang, rP, rQ = angles_deg(P, Q)
    Rad = np.abs(rP[:, None] - rQ[None, :])

    with np.errstate(invalid='ignore'):
        AE = Ang / float(threshold_deg)  # normalized angular error
        RE = Rad / float(threshold_r)    # normalized radial error

    C = np.maximum(AE, RE)
    C[~np.isfinite(C)] = big_cost  # invalid pairs -> huge cost

    # Optional clipping, above lambda so that bad matches are not hidden
    if cap_clip is not None:
        C = np.minimum(C, float(cap_clip))

    # Pad with dummies to allow unmatched elements
    n_rows, n_cols = C.shape
    L = max(n_rows, n_cols)

    C_full = np.full((L, L), big_cost, dtype=float)
    C_full[:n_rows, :n_cols] = C

    # Dummy columns: leave P elements unmatched
    if n_cols < L:
        C_full[:n_rows, n_cols:L] = float(lambda_unmatch)

    # Dummy rows: leave Q elements unmatched
    if n_rows < L:
        C_full[n_rows:L, :n_cols] = float(lambda_unmatch)

    # Dummy-dummy area, no influence
    if n_rows < L and n_cols < L:
        C_full[n_rows:L, n_cols:L] = 0.0

    # Assignment
    full_row, full_col = linear_sum_assignment(C_full)

    # Keep only real matches, inside the P × Q block
    keep = (full_row < n_rows) & (full_col < n_cols)

    row_ind = full_row[keep]
    col_ind = full_col[keep]

    # Quality filter / bad-pair detection relative to thresholds
    c_pairs = C[row_ind, col_ind]
    idx_bad = np.nonzero(c_pairs > 1.0)[0]

    # Euclidean distances
    if row_ind.size:
        dists = np.linalg.norm(P[row_ind] - Q[col_ind], axis=1)
    else:
        dists = np.array([], dtype=float)

    # Q reordered according to P, NaN for unmatched elements
    Q_reordered = np.full((N, D), np.nan, dtype=float)

    if row_ind.size:
        Q_reordered[row_ind] = Q[col_ind]

    return row_ind, col_ind, dists, Q_reordered, idx_bad