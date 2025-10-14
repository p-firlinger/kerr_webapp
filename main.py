import numpy as np
from src.calculations import KerrMetric
from src.integration import GeodesicIntegrator
from src.observer import Observer
from src.visualization import Visualizer
from src. shadow_visualization import ShadowCalculator
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def main():
    """
    Asks for user input and performs either 3D visualization, shadow visualization or data acquisition
    """

    st.set_page_config(page_title="Kerr Geodesic Visualizer", layout="wide")
    st.title("🌀 Geodesic Explorer in Kerr Spacetime")
    st.markdown("""
    **Explore the paths of light particles (photons) near a spinning ("Kerr"-) black hole!**  
    This tool lets you adjust initial conditions (spherical position and momentum components) and see how light behaves in Kerr space-time.  
    No advanced physics knowledge needed — just experiment with the sliders and watch the trajectories.
    """)
    # Sidebar inputs
    st.sidebar.header("Initial Conditions & Black Hole Parameters")
    r0 = st.sidebar.slider("Initial r", 1.0, 50.0, 3.0)
    theta0 = st.sidebar.slider("Initial θ (deg)", 0.0, np.pi, 90.0)
    phi0 = st.sidebar.slider("Initial φ (deg)", 0.0, 2*np.pi, 0.0)
    p_r0 = st.sidebar.slider("Initial p_r", -10.0, 10.0, 0.0)
    p_theta0 = st.sidebar.slider("Initial p_θ", -10.0, 10.0, 0.0)
    p_phi0 = st.sidebar.slider("Initial p_φ", -10.0, 10.0, -2.0)
    a = st.sidebar.slider("Spin parameter a", 0.0, 0.99, 0.0)
    M = st.sidebar.slider("Mass (M)", 1.0, 100.0, 1.0)
    lambda1 = st.sidebar.slider("Affine parameter range λ",100.0, 50.0)
    axis_lim = st.sidebar.slider("Axis limit (zoom)",0, 100, 10)
    if st.sidebar.button("Run Geodesic Integration"):
        with st.spinner("Computing geodesic..."):
            kerr = KerrMetric(M, a)
            lambda_span = (0, lambda1)   
            S0 = np.array([0, r0, theta0, phi0, p_r0, p_theta0, p_phi0])
            integrator = GeodesicIntegrator(kerr, M)
            solution = integrator.integrate(S0, lambda_span, 0.01)
            visualizer = Visualizer(M, a)
            fig = visualizer.plot_geodesics((solution.y[1], solution.y[2], solution.y[3]), axis_lim)
            st.pyplot(fig)

if __name__ == "__main__":
    main()