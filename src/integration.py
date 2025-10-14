import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from src.calculations import KerrMetric
from typing import Tuple


class GeodesicIntegrator:
    """
    Integrates geodesics for a given metric and initial conditions using Runge-Kutta iterative integration methods
    """

    def __init__(self, metric: KerrMetric, M: float):
        """
        Initialize metric and mass parameter of the black hole spacetime as class variables
        """
        self.metric = metric
        self.M = M

    def geodesic_equation(self, lambda_: float, S: np.ndarray) -> np.ndarray:
        """
        Implements the geodesic equation dS/dlambda = f(S)

        Inputs:
            lambda_:    (dummy variable for integration)
            S:  np.ndarray (array with 7 entries containing position and spatial momentum components)
        
        Returns:
            dS_dlambda: np.ndarray (expression for dS/dlambda as obtained by the geodesic equations using the null condition for p0)
        """
        x, p_spatial = S[:4], S[4:]  
        dS_dlambda = np.zeros(7)
        r, theta, phi = x[1], x[2], x[3]

        if r>100*self.M: #if r goes to infinity, stop the integration
            return np.full_like(S,np.nan)
            
        #compute p_0 from the null condition
        try:
            p_0 = self.metric.compute_p0(r, theta, p_spatial[0], p_spatial[1], p_spatial[2], "negative")
        except ValueError as e:
            return np.full_like(S, np.nan)  #return NaN to force solver to stop

        p_full = np.array([p_0, p_spatial[0], p_spatial[1], p_spatial[2]])

        try:
            Gamma = self.metric.christoffel_symbols(r, theta)
        except ValueError:
            return np.full_like(S, np.nan)
    
        metric_inv = self.metric.metric_inverse(r,theta)

        #compute dx^m/dlambda = p^m
        dS_dlambda[:4] = np.einsum("ij,j->i", metric_inv, p_full)

        #compute dp_m/dlambda = Gamma^a_{m l} p_a p^l
        for mu in range(1, 4):
            dp_mu = sum(
                Gamma[alpha][mu][lam] * p_full[alpha] * np.dot(metric_inv, p_full)[lam]
                for alpha in range(4) for lam in range(4))
        
            dS_dlambda[mu + 3] = dp_mu 

        return dS_dlambda
    
    
    def integrate(self, S0: np.ndarray, lambda_span: Tuple[float,float], max_step_=0.1):
        """
        Integrates the geodesic for given initial conditions S0 from lambda0 to lambda1 with a given maximal step size

        Inputs:
            S0: np.ndarray (S array for initial conditions)
            lambda_span:    Tuple[float,float] (initial and final boundaries for affine parameter)
            max_step_:  float (maximum step size for integrator)

        Returns:
            sol:    solve_ivp.OdeResult (sol.t gives a 1D array for affine parameter, sol.y a 2D array of S for each step, where each row is a fixed lambda)
        """

        sol = solve_ivp(
            self.geodesic_equation,
            lambda_span,
            y0 = S0,
            method = "DOP853",
            rtol=1e-8,
            atol=1e-8,
            max_step = max_step_,
            events=self.stop_at_horizon
        )
        return sol

    def stop_at_horizon(self, lambda_: float, S: np.ndarray) -> float:
        """
        Stops just befor the event horizon

        Inputs:
            lambda_:    float (dummie variable for affine parameter)
            S:  np.ndarray (seven dimensional state vector)

        Returns:
            distance:   float (distance between horizon and radial position, stricter condition by stopping already if it is 0.1)
        """
        r = S[1]
        distance = r - (self.metric.event_horizon() + 0.1)
        return distance
    
    stop_at_horizon.terminal = True #stop integration if this is triggered
    stop_at_horizon.direction = -1 #check if position is moving inward (i.e. distance is getting smaller)
    
    