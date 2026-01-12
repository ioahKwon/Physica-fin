"""
Colormap functions for torque visualization.
"""

import numpy as np


def torque_to_color_plasma(magnitude: float, max_torque: float = 300.0) -> tuple:
    """
    Convert torque magnitude to plasma colormap color (purple -> yellow).

    Args:
        magnitude: Torque magnitude in Nm
        max_torque: Maximum torque for normalization (default 300 Nm)

    Returns:
        RGB tuple (r, g, b) with values 0-1
    """
    if magnitude < 0.01:
        return (0.8, 0.8, 0.8)  # Light gray for no torque

    # Log scale normalization
    t = np.log10(magnitude + 1) / np.log10(max_torque + 1)
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


# Bright neon colors for axis visualization
AXIS_COLORS = {
    'x': (1.0, 0.3, 0.3),  # Bright Red/Pink
    'y': (0.3, 1.0, 0.3),  # Bright Green (neon)
    'z': (0.3, 0.6, 1.0),  # Bright Cyan/Blue
}
