import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple
from matplotlib.animation import FuncAnimation
from io import BytesIO
import streamlit as st
from matplotlib.animation import PillowWriter
import tempfile
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import streamlit.components.v1 as components

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
    def animate_geodesic(self, geodesic_data, axis_limit=20, max_frames=50):
        r, theta, phi = geodesic_data

        # Subsample points
        step = max(1, len(r)//max_frames)
        r_sub, theta_sub, phi_sub = r[::step], theta[::step], phi[::step]

        x = r_sub*np.sin(theta_sub)*np.cos(phi_sub)
        y = r_sub*np.sin(theta_sub)*np.sin(phi_sub)
        z = r_sub*np.cos(theta_sub)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot horizons
        r_plus = self.M + np.sqrt(self.M**2 - self.a**2)
        r_minus = self.M - np.sqrt(self.M**2 - self.a**2)
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        for r_horizon, color in zip([r_plus, r_minus], ["black", "gray"]):
            x_h = r_horizon * np.outer(np.cos(u), np.sin(v))
            y_h = r_horizon * np.outer(np.sin(u), np.sin(v))
            z_h = r_horizon * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_h, y_h, z_h, color=color, alpha=0.5)

        # Axis limits
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Line and dot
        line = Line3D([], [], [], color="darkred", lw=2)
        dot = Line3D([], [], [], marker='o', color='black', markersize=6)

        ax.add_line(line)
        ax.add_line(dot)
        def update(frame):
            line.set_data(x[:frame], y[:frame])
            line.set_3d_properties(z[:frame])
            dot.set_data([x[frame-1]], [y[frame-1]])
            dot.set_3d_properties(z[frame-1])
            return line, dot

        # Create animation
        anim = FuncAnimation(fig, update, frames=range(1, len(x)), interval=50)

        # Render in Streamlit
        html_anim = anim.to_jshtml()
        components.html(html_anim, width=800, height=800)