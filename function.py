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
    min_size_x=2.5, max_size_x=5, min_size_y=3, max_size_y=5, min_size_z=3, max_size_z=5
):
    # Génère aléatoirement les dimensions de la salle (L, W, H)
    L = np.random.uniform(min_size_x, max_size_x)
    W = np.random.uniform(min_size_y, max_size_y)
    H = np.random.uniform(min_size_z, max_size_z)
    return L, W, H


def generate_random_points(L, W, H, distance, distance_murs=1.5):
    # Assurer que la salle est assez grande pour placer deux points à une distance donnée
    if min(L, W, H) <= distance:
        raise ValueError(
            "La salle est trop petite pour placer les points à la distance spécifiée."
        )

    # Placer le premier point de manière aléatoire dans la salle
    point1 = np.array(
        [
            np.random.uniform(distance, L - distance),
            np.random.uniform(distance, W - distance),
            np.random.uniform(distance, H - distance),
        ]
    )

    # Placer le deuxième point à la distance donnée du premier point
    # Choisir une direction aléatoire pour le deuxième point
    theta = np.random.uniform(0, 2 * np.pi)  # Angle dans le plan XY
    phi = np.random.uniform(0, np.pi)  # Angle dans le plan Z
    x2 = point1[0] + distance * np.sin(phi) * np.cos(theta)
    y2 = point1[1] + distance * np.sin(phi) * np.sin(theta)
    z2 = point1[2] + distance * np.cos(phi)

    # Vérifier que le deuxième point est dans la salle et à une distance suffisante des bords
    if not (
        distance_murs <= x2 <= L - distance_murs
        and distance_murs <= y2 <= W - distance_murs
        and distance_murs <= z2 <= H - distance_murs
    ):
        return generate_random_points(
            L, W, H, distance, distance_murs
        )  # Si le point est trop proche des bords, refaire

    point2 = np.array([x2, y2, z2])
    return point1, point2


def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords

    # Calcul de la distance radiale
    r = np.sqrt(x**2 + y**2 + z**2)

    # Calcul de l'azimut (angle dans le plan XY)
    theta = np.arctan2(y, x)  # Utilise np.arctan2 pour éviter les problèmes de signes

    # Calcul de l'élévation (angle avec l'axe Z)
    phi = np.arccos(z / r) if r != 0 else 0  # Si r = 0, on évite une division par zéro

    # Conversion des angles en degrés
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    return r, theta_deg, phi_deg


def random_angles():
    rand_i, rand_j = np.random.rand(
        2
    )  # Two independent random numbers from a uniform distribution in the range (0, 1)
    theta = 2 * np.pi * rand_i  # Spherical coordinate theta
    phi = np.arccos(
        2 * rand_j - 1
    )  # Spherical coordinate phi, corrected for distribution bias
    return (np.degrees(theta), np.degrees(phi))


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
