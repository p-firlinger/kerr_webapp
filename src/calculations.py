import numpy as np
from scipy.integrate import solve_ivp

class KerrMetric:
    """
    Calculate the metric and other values for Kerr spacetime for given parameters M and a and at a given position (r,theta) in Boyer-Lindquist coordinates.
    """

    def __init__(self, M: float, a: float):
        """
        Initialize mass M and spin a of the black hole
        """
        self.M = M
        self.a = a

    def metric(self, r:float, theta: float) -> np.ndarray:
        """
        Returns the metric as a numpy array for a given position (r,theta) in Kerr spacetime.

        Inputs:
            r:  float (radial coordinate in Boyer-Lindquist coordinates)
            theta:  float (elevation coordinate in Boyer-Lindquist coordinates)
        
        Returns:
            g:  np.ndarray (metric tensor components in matrix representation)
        """
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        Sigma = r**2 + self.a**2 * cos_theta**2
        Delta = r**2 - 2*self.M*r + self.a**2
        sin_theta2 = sin_theta**2
        a2 = self.a**2

        g00 = -1* (Delta - a2 * sin_theta2)/Sigma
        g03 = -2 * self.M * r * self.a *sin_theta2/ Sigma
        g11 = Sigma / Delta
        g22 = Sigma
        g33 = ((r**2 + a2)**2 - a2 * Delta* sin_theta2)* sin_theta2 / Sigma

        g = np.array([
            [g00, 0, 0, g03],
            [0, g11, 0, 0],
            [0, 0, g22, 0],
            [g03, 0, 0, g33]])
        
        return g
    
    def metric_inverse(self, r: float, theta: float) -> np.ndarray:
        """
        Returns the inverse metric as a numpy array for a given position (r,theta) in Kerr spacetime

        Inputs:
            r:  float (radial coordinate in Boyer-Lindquist coordinates)
            theta:  float (elevation coordinate in Boyer-Lindquist coordinates)
        
        Returns:
            metric_inv:  np.ndarray (inverse metric tensor components in matrix representation)
        """
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        Sigma = r**2 + self.a**2 * cos_theta**2
        Delta = r**2 - 2*self.M*r + self.a**2
        a2 = self.a**2
        sin_theta2 = sin_theta**2
        SigmaDelta = Sigma*Delta

        g00 = -((r**2 + a2)**2 - a2 * Delta * sin_theta2) / (SigmaDelta)
        g03 = -2 * self.a * self.M * r / (SigmaDelta)
        g11 = Delta / Sigma
        g22 = 1 / Sigma
        g33 = (Delta - a2 * sin_theta2) / (SigmaDelta * sin_theta2)

        metric_inv = np.array([
            [g00, 0, 0, g03],
            [0, g11, 0, 0],
            [0, 0, g22, 0],
            [g03, 0, 0, g33]])
        
        return metric_inv
    
    def compute_p0(self, r: float, theta: float, p_r: float, p_theta: float, p_phi: float, time_direction = "positive") -> float:
        """
        Returns the four-momentums time-component as calculated from the spatial components making use of the fact that for photons, four-momentum is a null-vector.
        
        Inputs: 
            r:  float (radial coordinate in Boyer-Lindquist coordinates)
            theta:  float (elevation coordinate in Boyer-Lindquist coordinates)
            p_r:    float (radial momentum component)
            p_theta:    float (elevation momentum component)
            p_phi:  float (azimuthal momentum component)
            time_direction: str ("negative" for past-directed momentum four vector, "positive" for futer-directed)

        Returns:
            p0: float (time component of momentum four vector)
        """
        g = self.metric_inverse(r,theta)

        term1 = -1* g[0][3] * p_phi
        discriminant =   (g[0][3] * p_phi)**2 -  g[0][0] * (g[1][1] * p_r**2 + g[2][2] * p_theta**2 + g[3][3] * p_phi**2)
        term2 = np.sqrt(discriminant)

        if time_direction.lower() == "positive":
            p0 = (term1 + term2)/(g[0][0])
        elif time_direction.lower() == "negative":
            p0 = (term1 - term2) / (g[0][0]) 
        else:
            raise ValueError("Choose either \"positive\" or \"negative\" as a time-direction.")

        return p0
    
    def christoffel_symbols(self, r: float, theta: float) -> np.ndarray:
        """
        Returns the Christoffel symbols Gamma^alpha _{beta gamma} as a numpy array

        Inputs:
            r:  float (radial coordinate in Boyer-Lindquist coordinates)
            theta:  float (elevation coordinate in Boyer-Lindquist coordinates)
        
        Returns:
            christoffel:    np.ndarray (three-index np array that contains components of Christoffel symbols in Boyer-Lindquist coordinates)
        """
        a2 = self.a**2
        r_s  = 2*self.M
        r2 = r**2
        cos_theta, sin_theta = np.cos(theta), np.sin(theta) 
        cos_theta2 = cos_theta**2
        sin_theta2 = sin_theta**2
        Sigma = r**2 + self.a**2 * cos_theta2
        Sigma2 = Sigma**2
        Sigma3 = Sigma**3
        Delta = r2 - r_s * r + a2
        Sigma2Delta = Sigma2 * Delta
        SigmaDelta = Sigma*Delta
        sincos = sin_theta * cos_theta

        #avoid singularities
        if abs(Sigma) < 1e-9 or abs(Delta) < 1e-9 or abs(sin_theta) < 1e-9:
            raise ValueError("Singularity in Christoffel symbol calculation")
        
        christoffel = np.zeros((4,4,4)) 
        #deffine the Christoffel symbols as seen in the cited literature
        christoffel[1][0][0] = r_s * Delta * (r2 - a2 * cos_theta2)/(2*Sigma3)
        christoffel[0][1][0] = christoffel[0][0][1] = r_s * (r2 + a2) * (r2 - a2 * cos_theta2)/(2*Sigma2Delta)
        christoffel[0][0][2] = christoffel[0][2][0] = -r_s * a2 * r * sincos/(Sigma2)
        christoffel[1][0][3] = christoffel[1][3][0] = -Delta *r_s * self.a * sin_theta2 * (r2 - a2 * cos_theta2)/(2* Sigma3)
        christoffel[1][1][1] = (2*r* a2 *sin_theta2 - r_s * (r2 - a2 * cos_theta2))/(2*SigmaDelta)
        christoffel[1][1][2] = christoffel[1][2][1] = -a2 * sincos/Sigma
        christoffel[1][2][2] = -r*Delta/Sigma
        christoffel[3][2][3] = christoffel[3][3][2] =(cos_theta/sin_theta)*(Sigma2 + r_s * a2 * r * sin_theta2)/(Sigma2)
        christoffel[2][0][0] = -r_s * a2  * r * sincos / (Sigma3)
        christoffel[3][0][1] = christoffel[3][1][0] = r_s * self.a * (r2 - a2 * cos_theta2)/(2*Sigma2Delta)
        christoffel[3][0][2] = christoffel[3][2][0] = -r_s * self.a * r * (cos_theta/sin_theta) / (Sigma2)
        christoffel[2][0][3] = christoffel[2][3][0] = r_s * self.a * r * (r2 + a2 ) * sin_theta * cos_theta / (Sigma3)
        christoffel[2][1][1] = a2 * sincos / (SigmaDelta)
        christoffel[2][1][2] = christoffel[2][2][1] = r/Sigma
        christoffel[2][2][2] = -a2 * sincos / Sigma
        christoffel[0][2][3] = christoffel[0][3][2] = r_s * (self.a**3) * r * (sin_theta**3) * cos_theta / ( Sigma2)
        christoffel[0][1][3] = christoffel[0][3][1] = r_s * self.a * sin_theta2 * (a2 * cos_theta2 * (a2 - r2) - r2*(a2 + 3* r2)) / (2* Sigma2Delta)
        christoffel[3][1][3] = christoffel[3][3][1] = (2*r*Sigma2 + r_s * (self.a**4 * sin_theta2 * cos_theta2 - r2 * (Sigma + r2 + a2 )))/(2*Sigma2Delta)
        christoffel[1][3][3] = Delta * sin_theta2 *(-2*r*Sigma2 + r_s * a2 * sin_theta2 * (r2 - a2 * cos_theta2))/(2*Sigma3)
        christoffel[2][3][3] = -1 * sin_theta * cos_theta * (((r2 + a2)**2 - a2 * Delta * sin_theta2) * Sigma + (r2 + a2)*r_s * a2 *r * sin_theta2)/(Sigma3)

        return christoffel
    
    def event_horizon(self) -> float:
        """
        Returns the radius of the outer event horizon for a Kerr black hole of given M and a.

        Returns:
            horizon:    float (radius of outer event horizon)
        """
        horizon = self.M + np.sqrt(self.M**2 - self.a**2)
        return horizon
