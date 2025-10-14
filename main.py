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
    st.title("Geodesic Explorer in Kerr Spacetime")
    st.markdown("""
    **Explore the paths of light particles (photons) near a spinning ("Kerr"-) black hole!**  
    This tool lets you adjust initial conditions (spherical position and momentum components) and see how light behaves in Kerr space-time.  
    No advanced physics knowledge needed — just experiment with the sliders and watch the trajectories.
    """)
    # Sidebar inputs
    st.sidebar.header("Initial Conditions & Black Hole Parameters")
    r0 = st.sidebar.slider("Initial r", 1.0, 50.0, 3.0)
    theta0 = st.sidebar.slider(r"Initial $\theta$ (deg)", 0.0, np.pi, np.pi/2)
    phi0 = st.sidebar.slider(r"Initial $\phi$ (deg)", 0.0, 2*np.pi, 0.0)
    p_r0 = st.sidebar.slider(r"Initial $p_r$", -5.0, 5.0, 0.0)
    p_theta0 = st.sidebar.slider(r"Initial $p_\theta$", -5.0, 5.0, 0.0)
    p_phi0 = st.sidebar.slider(r"Initial $p_\phi$", -5.0, 5.0, -2.0)
    a = st.sidebar.slider(r"Spin parameter $a$", -0.99, 0.99, 0.0)
    M = st.sidebar.slider(r"Mass ($M$)", 1.0, 100.0, 1.0)
    lambda1 = st.sidebar.slider(r"Affine parameter range $\lambda$", 10.0, 50.0)
    axis_lim = st.sidebar.slider(r"Axis limit (zoom)", 0, 50, 5)
    
    if st.sidebar.button("Run Geodesic Animation"):
        with st.spinner("Computing geodesic and preparing visualization..."):
            # Integrate the geodesic
            kerr = KerrMetric(M, a)
            lambda_span = (0, lambda1)
            S0 = np.array([0, r0, theta0, phi0, p_r0, p_theta0, p_phi0])
            integrator = GeodesicIntegrator(kerr, M)
            solution = integrator.integrate(S0, lambda_span, 0.01)

            r, theta, phi = solution.y[1], solution.y[2], solution.y[3]
            # Subsample if too long
            max_frames = 100
            step = max(1, len(r)//max_frames)
            r, theta, phi = r[::step], theta[::step], phi[::step]

            # Convert to Cartesian
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)

            # 1️⃣ Create figure
            fig = go.Figure()

            # 2️⃣ Add horizons (they are fixed, no animation)
            r_plus = M + np.sqrt(M**2 - a**2)
            r_minus = M - np.sqrt(M**2 - a**2)
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            for r_horizon, color in zip([r_plus, r_minus], ['black', 'gray']):
                X = r_horizon * np.outer(np.cos(u), np.sin(v))
                Y = r_horizon * np.outer(np.sin(u), np.sin(v))
                Z = r_horizon * np.outer(np.ones_like(u), np.cos(v))

                fig.add_trace(go.Mesh3d(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    i=[i for i in range((len(u)-1)*(len(v)-1))],  # simple triangulation
                    j=[i+1 for i in range((len(u)-1)*(len(v)-1))],
                    k=[i+len(v) for i in range((len(u)-1)*(len(v)-1))],
                    color='black',
                    opacity=1
                ))

            # 3️⃣ Add geodesic line (fixed)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='darkred', width=5),
                name='Photon path'
            ))

            # 4️⃣ Add moving photon dot
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                marker=dict(size=10, color='black'),
                name='Photon'
            ))
            photon_trace_index = len(fig.data) - 1  

            # 5️⃣ Build frames
            frames = [
                go.Frame(
                    data=[go.Scatter3d(
                        x=[x[i]], y=[y[i]], z=[z[i]],
                        mode='markers',
                        marker=dict(size=10, color='black')
                    )],
                    name=str(i),
                    traces=[photon_trace_index]
                )
                for i in range(len(x))
            ]

            fig.frames = frames

            # 6️⃣ Layout and animation buttons
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-axis_lim, axis_lim]),
                    yaxis=dict(range=[-axis_lim, axis_lim]),
                    zaxis=dict(range=[-axis_lim, axis_lim]),
                    aspectmode='cube'
                ),
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor='left',
                    yanchor='bottom',
                    buttons=[dict(
                        label='Play',
                        method='animate',
                        args=[None, {"frame": {"duration":50, "redraw":True}, "fromcurrent":True}]
                    )]
                )]
            )


            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()