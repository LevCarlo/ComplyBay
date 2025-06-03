import numpy as np
from scipy.interpolate import CubicSpline

class VpEstimate:
    def __init__(self, z=None, vs=None, rho=None):
        self.z = z
        self.vs = vs
        self.rho = rho
    
    def estimate(self, keyword, **kwargs): 
        keyword_map = {
            "Brocher2005": self._brocher,
            "Castegna1995": self._castegna,
            "Christensen&Shaw1970": self._christensen_shaw,
        }
        
        if keyword in keyword_map:
            return keyword_map[keyword]()
        
        elif keyword == "VpVsConst":
            value = kwargs.get("value", 1.73)
            if value is None:
                raise ValueError("VpVsConst requires Vp2Vs to be set.")
            return self.vs * value

        elif keyword == "VpConst":
            value = kwargs.get("value", 5.8)
            if value is None:
                raise ValueError("VpConst requires Vp to be set.")
            return np.full_like(self.vs, value)
        else:
            raise ValueError(f"Unknown keyword: {keyword}")
    
    def _brocher(self):
        conditions = np.logical_and(0 <= self.vs, self.vs <= 4.5)
        return np.where(
            conditions,
            0.9409 + 2.0947 * self.vs - 0.8206 * self.vs ** 2 + 0.2683 * self.vs ** 3 - 0.0251 * self.vs ** 4,
            np.nan
        )
    
    def _castegna(self):
        return 1.16 * self.vs + 1.36
    
    def _christensen_shaw(self):
        return 1.511 + 1.304 * self.z - 0.741 * self.z ** 2 + 0.257 * self.z ** 3
    

class RhoEstimate:
    def __init__(self, z=None, vs=None, vp=None):
        self.z = z
        self.vs = vs
        self.vp = vp
    
    def estimate(self, keyword, **kwargs):
        keyword_map = {
            "Brocher2005": self._brocher,
            "Ludwig1979": self._ludwig,
            "Gardner1974": self._gardner,
            "Christensen&Mooney1995": self._christensen_mooney,
            "Godfrey1997": self._godfrey,
            "Feng1986": self._feng,
            "Willoughby&Edwards2000": self._willoughby_edwards,
            "Hamilton1979": self._hamilton
        }

        if keyword in keyword_map:
            return keyword_map[keyword]()
        elif keyword == "RhoConst":
            value = kwargs.get("value", 2.0)
            if value is None:
                raise ValueError("RhoConst requires Rho to be set.")
            return np.full_like(self.vs, value)
        else:
            raise ValueError(f"Unknown keyword: {keyword}")
        
    def _brocher(self):
        return 1.22679 + 1.53201 * self.vs - 0.83668 * self.vs ** 2 + 0.20673 * self.vs ** 3 - 0.01656 * self.vs ** 4
    
    def _ludwig(self):
        conditions = np.logical_and(1.5 <= self.vp, self.vp <= 8.5)
        return np.where(
            conditions,
            1.6612 * self.vp - 0.4721 * self.vp ** 2 + 0.0671 * self.vp ** 3 - 0.0043 * self.vp ** 4 +
            0.000106 * self.vp ** 5,
            np.nan
        )
    def _gardner(self):
        conditions = np.logical_and(1.5 <= self.vp, self.vp <= 6.1)
        return np.where(
            conditions,
            1.74 * self.vp ** 0.25,
            np.nan
        )
    
    def _christensen_mooney(self):
        conditions = np.logical_and(5.5 <= self.vp, self.vp <= 7.5)
        return np.where(
            conditions,
            0.541 + 0.3601 * self.vp,
            np.nan
        )
    
    def _godfrey(self):
        conditions = np.logical_and(5.9 <= self.vp, self.vp <= 7.1)
        return np.where(
            conditions,
            2.4372 + 0.0761 * self.vp,
            np.nan
        )

    def _feng(self):
        conditions = [
            np.logical_and(self.vp <= 6, np.isnan(self.vp) == False),
            np.logical_and(self.vp > 6, self.vp < 7.5),
            np.logical_and(self.vp >= 7.5, np.isnan(self.vp) == False)
        ]
        choices = [
            2.78 + 0.56 * (self.vp - 6.0),
            3.07 + 0.29 * (self.vp - 7.0),
            3.22 + 0.20 * (self.vp - 7.5)
        ]
        return np.select(conditions, choices, default=np.nan)
    
    def _willoughby_edwards(self):
        return 1.7 + (0.2 / 300) * self.z

    def _hamilton(self):
        return 1.85 + 0.165 * self.vp

def layer2profile(thickness, values, z=None, interp_method="cubic"):
    """
    Convert layer values to a profile with depth.
    
    Parameters:
    - thickness: array of layer thicknesses
    - values: array of values for each layer
    - z: optional array of depths, if None, will be calculated
    
    Returns:
    - z: array of depths
    - profile: array of values at each depth
    """
    edges = np.concatenate([[0], np.cumsum(thickness)])
    centers = edges[:-1] + np.diff(edges) / 2

    if z is None:
        z = np.linspace(0, edges[-1], len(centers) + 1)
    
    if interp_method == "cubic":
        cs = CubicSpline(centers, values, bc_type='natural', extrapolate=True)
        fz = cs(z)
    elif interp_method == "linear":
        fz = np.interp(z, centers, values)
    else:
        raise ValueError(f"Unknown interpolation method: {interp_method}")
    return fz
    



if __name__ == "__main__":
    # Example usage
    z = np.array([0, 0.6, 1.5, 3.0])  # Depth in km
    vs = np.array([1.5, 2.0, 2.5, 3.0])

    vp_estimator = VpEstimate(z=z, vs=vs)
    rho_estimator = RhoEstimate(z=z, vs=vs)
    vp_brocher = vp_estimator.estimate("Brocher2005")
    rho_brocher = rho_estimator.estimate("Brocher2005")

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
    # ax[0].plot(vs, z, label='Vs')
    # ax[1].plot(vp_brocher, z, label='Vp (Brocher 2005)')
    # ax[2].plot(rho_brocher, z, label='Rho (Brocher 2005)')
    # ax[0].invert_yaxis()
    # # plt.show()
    # fig.savefig