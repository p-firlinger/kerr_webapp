import numpy as np
from src.calculations import KerrMetric
class Observer:
    """
    Represents a resting observer in Boyer-Lindquist coordinates and assigns a Vierbein
    """

    def __init__(self, M: float, a: float, r: float, theta: float, phi: float, metric: KerrMetric):
        """
        Initialize mass M, spin a, spatial Boyer Lindquist coordinates and metric for Kerr spacetime, as well as tetrad for static observer
        """
        self.M = M
        self.a = a
        self.r = r
        self.theta = theta
        self.phi = phi
        self.metric = metric
        self.tetrad_matrix = self.tetrad()

    def tetrad(self) -> np.ndarray:
        """
        Returns the tetrad as a matrix for a static observer at a given position
    
        Returns:
            tetrad: np.ndarray (tetrad for fixed observer, stacked four vectors)
        """
        Sigma = self.r**2 + self.a**2 *np.cos(self.theta)**2
        Delta = self.r**2 - 2*self.M*self.r + self.a**2 
        Chi = self.a* np.sin(self.theta)**2

        e0 = 1/np.sqrt(Sigma * Delta) * (self.a*np.array([0,0,0,1]) + (Sigma + self.a*Chi) *np.array([1,0,0,0]))
        e1 = -np.sqrt(Delta/Sigma) * np.array([0,1,0,0])
        e2 = np.sqrt(1/Sigma)*np.array([0,0,1,0])
        e3 = 1/(np.sqrt(Sigma)*np.sin(self.theta)) * (np.array([0,0,0,1])+ Chi*np.array([1,0,0,0]))
        tetrad = np.vstack([e0,e1,e2,e3]) 
        return tetrad

    def tetrad_to_coordinate(self, p_local: np.ndarray) -> np.ndarray:
        """
        Transforms Vierbein vector components to covector components in Boyer-Lindquist coordinate basis

        Inputs: 
            p_local:    np.ndarray (local components of four momentum)
        
        Returns:
            p_covariant:    np.ndarray (covariant momentum components in coordinate basis)
        """

        p_contravariant = np.zeros_like(p_local)
 
        for mu in range(4):
            for a in range(4):
                p_contravariant[mu] += p_local[a] * self.tetrad_matrix[a][mu]

        p_covariant = self.metric @ p_contravariant
        return p_covariant
    
    def null_condition(self, p_2, p_3):
        """
        Implement the Minkowski null condition (locally), choose p_0 = 1 without loss of generality because of the scale invariance of null geodesics
        
        Inputs:
            p_2:    float (elevation component of four momentum in the local frame)
            p_2:    float (azimuthal component of four momentum in the local frame)
        
        Returns:
            p_1:    float (radial component of four momentum in the local frame as calculated by the local null condition)
        """
        if p_2**2 + p_3**2 > 1:
            raise ValueError("Unphysical direction")
        p_1 = np.sqrt(1- p_2**2 - p_3**2)
        return p_1


    
    
