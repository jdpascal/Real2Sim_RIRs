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

import numpy as np


def fibonacci_sphere(samples=32, rds=0.042):
    """
      function for replacing the points on the Eigenmike' sphere, radius = 4.2cm and 32 mics

    Returns:
        points : list with coordinates of the mics on the sphere
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append(rds * np.array([x, y, z]))
        # points.append([])

    return np.array(points)


def fibonacci(samples=32, rds=0.042):
    """
    put samples on a sphere of radius rds

    Args:
        samples (int, optional): _description_. Defaults to 32.
        rds (int, optional): _description_. Defaults to 0.042 .
    """
    pts = []
    goldenRatio = (1 + 5**0.5) / 2
    for i in range(samples):
        theta = 2 * np.pi * i / goldenRatio
        phi = np.arccos(1 - 2 * (i + 0.5) / samples)
        x, y, z = (
            np.cos(theta) * np.sin(phi) * rds,
            np.sin(theta) * np.sin(phi) * rds,
            np.cos(phi) * rds,
        )
        pts.append(np.array([x, y, z]))

    return np.array(pts)


# murs entre 2,5 et 5 metres, plafond entre 3 et 5
def generate_random_room_dimensions(
    min_size_x=7.5, max_size_x=10.0, min_size_y=3.5, max_size_y=6.0, min_size_z=2.0, max_size_z=4.0,
):
    # Génère aléatoirement les dimensions de la salle (L, W, H)
    L = np.random.uniform(min_size_x, max_size_x)
    W = np.random.uniform(min_size_y, max_size_y)
    H = np.random.uniform(min_size_z, max_size_z)
    return L, W, H


# def generate_random_points(L, W, H, distance, distance_murs=1.5):
#     # Assurer que la salle est assez grande pour placer deux points à une distance donnée
#     if min(L, W, H) <= distance - distance_murs * 2:
#         raise ValueError("La salle est trop petite pour placer les points à la distance spécifiée.")
#     # Assurer que la distance des murs est respectée
#     if distance_murs >= min(L, W, H) / 2:
#         raise ValueError("La distance des murs est trop grande par rapport aux dimensions de la salle.")
#     # Placer le premier point de manière aléatoire dans la salle
#     point1 = np.array([np.random.uniform(distance_murs, L - distance_murs),
#                        np.random.uniform(distance_murs, W - distance_murs),
#                        np.random.uniform(distance_murs, H - distance_murs)])
    
#     # Placer le deuxième point à la distance donnée du premier point
#     point2 = np.array([np.random.uniform(distance_murs, L - distance_murs),
#                        np.random.uniform(distance_murs, W - distance_murs),
#                        np.random.uniform(distance_murs, H - distance_murs)])

#     while np.linalg.norm(point2 - point1) < distance:
#         # print("                 POINT 1:", point1, ", POINT 2:", point2, ", DISTANCE:", distance, ", NORM:", np.linalg.norm(point2 - point1))
#         point1 = np.array([np.random.uniform(distance_murs, L - distance_murs),
#                            np.random.uniform(distance_murs, W - distance_murs),
#                            np.random.uniform(distance_murs, H - distance_murs)])
#         point2 = np.array([np.random.uniform(distance_murs, L - distance_murs),
#                            np.random.uniform(distance_murs, W - distance_murs),
#                            np.random.uniform(distance_murs, H - distance_murs)])
#     point2 = point1 + (point2 - point1) * (distance / np.linalg.norm(point2 - point1))
    
#     return point1, point2

def generate_random_points(L, W, H, distance, distance_murs=1.5, max_point_attempts=80, max_direction_attempts=100):
    """Generate two points inside the room at a fixed Euclidean distance.

    The first point is placed randomly within the safe interior, then a
    random unit direction is sampled until the second point at the given
    distance is also inside the safe bounds.
    """
    min_bound = np.array([distance_murs, distance_murs, distance_murs], dtype=float)
    max_bound = np.array([L - distance_murs, W - distance_murs, H - distance_murs], dtype=float)

    if np.any(max_bound <= min_bound):
        raise ValueError("Les dimensions de la salle sont trop petites pour respecter la distance aux murs.")

    max_possible_distance = np.linalg.norm(max_bound - min_bound)
    if distance > max_possible_distance:
        raise ValueError(
            "La salle est trop petite pour placer deux points à la distance spécifiée. "
            f"Distance max possible = {max_possible_distance:.3f}, demandée = {distance:.3f}."
        )

    def random_unit_vector():
        v = np.random.normal(size=3)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

    def point_inside_bounds(point):
        return np.all(point >= min_bound) and np.all(point <= max_bound)

    for _ in range(max_point_attempts):
        point1 = np.random.uniform(min_bound, max_bound)
        for _ in range(max_direction_attempts):
            direction = random_unit_vector()
            point2 = point1 + direction * distance
            if point_inside_bounds(point2):
                return point1, point2

    # Fallback: try systematic axis-aligned solutions if random search fails.
    for _ in range(max_point_attempts):
        point1 = np.random.uniform(min_bound, max_bound)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                candidate = point1.copy()
                candidate[axis] = point1[axis] + sign * distance
                if point_inside_bounds(candidate):
                    return point1, candidate

    raise RuntimeError(
        "Impossible de générer deux points à la distance fixe pour ces dimensions de salle. "
        "Vérifiez la taille de la salle et la distance demandée."
    )
    


def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords
    
    # Calcul de la distance radiale
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Calcul de l'azimut (angle dans le plan XY)
    phi = np.arctan2(y, x)  # Utilise np.arctan2 pour éviter les problèmes de signes

    rho   = np.hypot(x, y)               # projection sur le plan XY
    theta = np.arctan2(z, rho)           # élévation
    # phi   = 0.0                          # roulis

    # Conversion des angles en degrés
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    
    return r, theta_deg, phi_deg

# def random_angles():
#     rand_i, rand_j = np.random.rand(
#         2
#     )  # Two independent random numbers from a uniform distribution in the range (0, 1)
#     theta = 2 * np.pi * rand_i  # Spherical coordinate theta
#     phi = np.arccos(
#         2 * rand_j - 1
#     )  # Spherical coordinate phi, corrected for distribution bias
#     return (np.degrees(theta), np.degrees(phi))


def random_angles():
    rand_x, rand_y, rand_z = np.random.rand(3) * 2 -1                  # three independent random numbers from a uniform distribution in the range (0, 1)
    theta, phi = cartesian_to_spherical([rand_x, rand_y, rand_z])[1:] # Spherical coordinate theta and phi
    return( theta, phi )


def approximation_distance(more_or_less=0.1):
    approx = np.random.rand()
    return 2 * more_or_less * approx - more_or_less  # approximation of more or less cm


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

P = np.random.uniform(abs_coeffs_lower_bound, abs_coeffs_upper_bound)
