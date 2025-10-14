import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple

class Visualizer:
    """
    Plot geodesics, horizons and shadows
    """
    def __init__(self, M: float, a: float):
        """
        Initialize mass M and spin a of spacetime
        """
        self.M = M
        self.a = a
    def plot_geodesics(self, geodesic_data: Tuple[np.ndarray, np.ndarray, np.ndarray], axis_limit: float):
        """
        Plots geodesics in 3D
        
        Inputs:
            geodesic_data:  Tuple[np.ndarray, np.ndarray, np.ndarray] (spatial coordinates of geodesic for each step of the affine parameter)
            axis_limit: float (maximally displayed value on the axis, i.e. "zoom")
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        #plot the horizons
        r_plus = self.M + np.sqrt(self.M**2 - self.a**2)
        r_minus = self.M - np.sqrt(self.M**2 - self.a**2)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        for r_horizon, color in zip([r_plus, r_minus], ["black", "gray"]):
            x_horizon = r_horizon * np.outer(np.cos(u), np.sin(v))
            y_horizon = r_horizon * np.outer(np.sin(u), np.sin(v))
            z_horizon = r_horizon * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_horizon, y_horizon, z_horizon, color=color, alpha=0.5)
        
        #plot geodesic (convert to coordinates for embedding into cartesian R3)
        r, theta, phi = geodesic_data

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        
        ax.plot(x, y, z, color="darkred", alpha=1)

        
        ax.set_xlim([-axis_limit,axis_limit])
        ax.set_ylim([-axis_limit,axis_limit])
        ax.set_zlim([-axis_limit,axis_limit])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel(r"$x \ [GM/c^2]$", fontsize=10)
        ax.set_ylabel(r"$y \ [GM/c^2]$", fontsize=10)
        ax.set_zlabel(r"$z \ [GM/c^2]$", fontsize=10)
        
        
        return fig 
