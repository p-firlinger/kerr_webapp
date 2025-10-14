import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from src.calculations import KerrMetric
from src.observer import Observer
from src.integration import GeodesicIntegrator
from typing import Tuple
import os
import pandas as pd
from tqdm import tqdm

class ShadowCalculator:
    """
    Computes and visualizes the shadow of a Kerr black hole as seen by a static observer in Boyer-Lindquist coordinates
    """

    def __init__(self, M: float, a: float):
        """
        Initialize mass M, spin a, metric and outer event horizon of the spacetime
        """
        self.M = M
        self.a = a
        self.metric = KerrMetric(M,a)
        self.horizon = M + np.sqrt(M**2 - a**2)

    def compute_shadow(self, r_obs: float, theta_obs: float, phi_obs: float, N: int = 20, tetrad_range: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes those local p2 and p3 initial directions that cause a photon to reach the event horizon (i.e. fall into the black hole)

        Inputs: 
            r_obs:  float (radial coordinate of stationary observer)
            theta_obs:  float (elevation coordinate of stationary observer)
            phi_obs:  float (azimuthal coordinate of stationary observer)
            N:  int (number of initial conditions to be computed)
            tetrad_range:   float (scope on celestial sphere that is covered, 1 corresponds to the whole semisphere pointing in the direction of the B.H.)
        
        Returns:
            (p2,p3):    Tuple[np.ndarray, np.ndarray] (initial p2 and p3 directions that cause the event horizon to be reached)
        """
        
        x_obs = np.array([0, r_obs, theta_obs, phi_obs])
        metric_obs = self.metric.metric(r_obs, theta_obs)
        observer = Observer(self.M, self.a, r_obs, theta_obs, phi_obs, metric_obs)
        local_tetrad = observer.tetrad()

        p_2 = np.linspace(-tetrad_range, tetrad_range, N)
        p_3 = np.linspace(-tetrad_range, tetrad_range, N)

        p_2_shadow, p_3_shadow = [], []

        integrator = GeodesicIntegrator(self.metric, self.M)

        for i in tqdm(range(N), desc="p2 loop"): #show progress in the command line
            for j in range(N):
                try:
                    p_1 = observer.null_condition(p_2[i], p_3[j])
                except ValueError:
                    continue

                p_local = np.array([1,p_1,p_2[i], p_3[j]])

                p_coord = observer.tetrad_to_coordinate(p_local)
                p_coord[0] = self.metric.compute_p0(r_obs, theta_obs, p_coord[1], p_coord[2], p_coord[3], time_direction="negative")
                
                S0 = np.array([0,r_obs, theta_obs, phi_obs, p_coord[1], p_coord[2], p_coord[3]])
                result = integrator.integrate(S0, (0,1e4), max_step_=0.5)

                if result.status == 1:
                    p_2_shadow.append(p_local[2])
                    p_3_shadow.append(p_local[3])

        return np.array(p_2_shadow), np.array(p_3_shadow)
    
    def plot_shadow(self, p_2_shadow: np.ndarray, p_3_shadow: np.ndarray):
        """
        Plots the directions that cause the photon to fall in as a scatter plot
        
        Inputs:
            p_2_shadow: np.ndarray (array of local elevation components)
            p_3_shadow: np.ndarray (array of local azimuthal components)
        """
        plt.figure(figsize=(6,6))
        plt.scatter(p_2_shadow, p_3_shadow, s=30, color="black")
        plt.xlabel(r"$p_2$")
        plt.ylabel(r"$p_3$")
        plt.title(f"Black Hole Shadow (a={self.a})")
        plt.axis('equal')
        plt.grid(True)
        plt.show()






    def generate_dataset(self, r_obs: float, theta_obs: float, phi_obs: float, N: float, tetrad_range: float, csv_name: str):
        """
        Generate relevant data for N^2 different photon directions for some specific observer position and append to csv file.
        """
        x_obs = np.array([0,r_obs, theta_obs, phi_obs])
        metric_obs = self.metric.metric(r_obs, theta_obs)
        observer = Observer(self.M, self.a, r_obs, theta_obs, phi_obs, metric_obs)
        p_2 = np.linspace(-tetrad_range, tetrad_range, N)
        p_3 = np.linspace(-tetrad_range, tetrad_range, N)
        integrator = GeodesicIntegrator(self.metric, self.M)

        data_rows = []

        for i in tqdm(range(N), desc="P2 loop"):
            for j in range(N):
                try:
                    p1 = observer.null_condition(p_2[i], p_3[j])
                except ValueError:
                    continue

                p_local = np.array([1,p1, p_2[i], p_3[j]])
                p_coord = observer.tetrad_to_coordinate(p_local)
                p_coord[0] = self.metric.compute_p0(r_obs, theta_obs, p_coord[1], p_coord[2], p_coord[3], time_direction="negative")

                S0 = np.array([0,r_obs, theta_obs, phi_obs, p_coord[1], p_coord[2], p_coord[3]])
                result = integrator.integrate(S0, (0,1e4), max_step_=0.5)

                if result.status == 1:
                    hit = 1
                else: 
                    hit = 0

                data_rows.append({
                    "M": self.M,
                    "a": self.a,
                    "r_obs": r_obs,
                    "theta_obs": theta_obs,
                    "phi_obs": phi_obs,
                    "p1": p1,
                    "p2": p_2[i],
                    "p3": p_3[j],
                    "hit": hit   
                                  })
        
        #create dataframe
        df = pd.DataFrame(data_rows)

        #append to csv file
        if os.path.exists(csv_name):
            df.to_csv(csv_name, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_name, index=False)
        

