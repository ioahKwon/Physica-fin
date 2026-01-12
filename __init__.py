"""
SKEL Force Visualization Package
Visualize joint torques on SKEL skeleton mesh with PhysPT-style rendering.
"""

from .colormaps import torque_to_color_plasma, torque_to_color_jet
from .mesh_utils import (
    create_line_mesh,
    create_sphere_mesh,
    read_obj_mesh,
    write_obj_with_color,
)
from .visualizer import SKELForceVisualizer

__version__ = "1.0.0"
__all__ = [
    "torque_to_color_plasma",
    "torque_to_color_jet",
    "create_line_mesh",
    "create_sphere_mesh",
    "read_obj_mesh",
    "write_obj_with_color",
    "SKELForceVisualizer",
]
