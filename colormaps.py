"""
Colormap functions for torque visualization.
"""

import numpy as np


def torque_to_color_plasma(magnitude: float, max_torque: float = 300.0, min_torque: float = 0.0) -> tuple:
    """
    Convert torque magnitude to plasma colormap color (purple -> yellow).

    Args:
        magnitude: Torque magnitude in Nm
        max_torque: Maximum torque for normalization
        min_torque: Minimum torque for normalization (default 0)

    Returns:
        RGB tuple (r, g, b) with values 0-1
    """
    if magnitude < 0.01:
        return (0.8, 0.8, 0.8)  # Light gray for no torque

    # Linear normalization using actual min/max range
    torque_range = max_torque - min_torque
    if torque_range < 0.1:
        torque_range = 1.0  # Avoid division by zero
    t = (magnitude - min_torque) / torque_range
    t = np.clip(t, 0, 1)

    # Plasma colormap control points (from matplotlib)
    colors = [
        (0.050383, 0.029803, 0.527975),   # 0.0 - dark purple
        (0.417642, 0.000564, 0.658390),   # 0.25 - purple
        (0.798216, 0.280197, 0.469538),   # 0.5 - magenta/pink
        (0.988362, 0.557937, 0.231441),   # 0.75 - orange
        (0.940015, 0.975158, 0.131326),   # 1.0 - yellow
    ]

    # Linear interpolation
    idx = t * 4
    i = int(idx)
    if i >= 4:
        return colors[4]
    frac = idx - i
    r = colors[i][0] * (1 - frac) + colors[i + 1][0] * frac
    g = colors[i][1] * (1 - frac) + colors[i + 1][1] * frac
    b = colors[i][2] * (1 - frac) + colors[i + 1][2] * frac
    return (r, g, b)


def torque_to_color_jet(magnitude: float, max_torque: float = 300.0) -> tuple:
    """
    Convert torque magnitude to jet colormap color (blue -> cyan -> green -> yellow -> red).

    Args:
        magnitude: Torque magnitude in Nm
        max_torque: Maximum torque for normalization (default 300 Nm)

    Returns:
        RGB tuple (r, g, b) with values 0-1
    """
    if magnitude < 0.01:
        return (0.8, 0.8, 0.8)  # Light gray for no torque

    # Log scale normalization
    log_val = np.log10(magnitude + 1) / np.log10(max_torque + 1)
    log_val = np.clip(log_val, 0, 1)

    # Jet colormap
    if log_val < 0.25:
        r = 0
        g = 4 * log_val
        b = 1
    elif log_val < 0.5:
        r = 0
        g = 1
        b = 1 - 4 * (log_val - 0.25)
    elif log_val < 0.75:
        r = 4 * (log_val - 0.5)
        g = 1
        b = 0
    else:
        r = 1
        g = 1 - 4 * (log_val - 0.75)
        b = 0

    return (r, g, b)


# Axis colors for torque arrows (Red/Blue/Yellow)
AXIS_COLORS = {
    'x': (0.9, 0.2, 0.2),  # Red
    'y': (0.2, 0.4, 0.9),  # Blue
    'z': (0.9, 0.8, 0.2),  # Yellow
}


def torque_to_color_dark_purple(magnitude: float, max_torque: float = 300.0) -> tuple:
    """
    Dark purple tone gradient for skeleton mesh coloring.

    - Low torque: very dark purple
    - High torque: bright lavender/purple

    Args:
        magnitude: Torque magnitude in Nm
        max_torque: Maximum torque for normalization (default 300 Nm)

    Returns:
        RGB tuple (r, g, b) with values 0-1
    """
    if magnitude < 0.01:
        return (0.12, 0.08, 0.18)  # Very dark purple for no torque

    # Log scale normalization
    t = np.log10(magnitude + 1) / np.log10(max_torque + 1)
    t = np.clip(t, 0, 1)

    # Dark purple gradient
    # t=0: very dark purple
    # t=1: bright lavender/purple
    dark = np.array([0.15, 0.08, 0.25])
    bright = np.array([0.65, 0.45, 0.85])

    color = dark + t * (bright - dark)
    return tuple(color)


def torque_to_color_green(magnitude: float, max_torque: float = 300.0) -> tuple:
    """
    Warm orange-yellow gradient for skeleton mesh coloring (Phys-SMPL style).

    - Low torque: dark red/orange
    - High torque: bright yellow

    Args:
        magnitude: Torque magnitude in Nm
        max_torque: Maximum torque for normalization (default 300 Nm)

    Returns:
        RGB tuple (r, g, b) with values 0-1
    """
    if magnitude < 0.01:
        return (0.106, 0.106, 0.129)  # Dark (#1B1B21) for no torque

    # Log scale normalization
    t = np.log10(magnitude + 1) / np.log10(max_torque + 1)
    t = np.clip(t, 0, 1)

    # Blue gradient (dark -> bright blue)
    # #1B1B21 = (27, 27, 33) = (0.106, 0.106, 0.129)
    # #0065FE = (0, 101, 254) = (0.0, 0.396, 0.996)
    colors = [
        np.array([0.106, 0.106, 0.129]),   # 0.0 - dark (#1B1B21)
        np.array([0.05, 0.20, 0.50]),      # 0.33 - dark blue
        np.array([0.0, 0.30, 0.80]),       # 0.66 - medium blue
        np.array([0.0, 0.396, 0.996]),     # 1.0 - bright blue (#0065FE)
    ]

    # Linear interpolation between control points
    idx = t * 3
    i = int(idx)
    if i >= 3:
        return tuple(colors[3])
    frac = idx - i
    color = colors[i] * (1 - frac) + colors[i + 1] * frac
    return tuple(color)
