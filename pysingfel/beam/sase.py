import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .base import Beam

class SASEBeam(Beam):
    def __init__(self, mu=None, sigma=None, n_spikes=0,
                 *args, **kargs):
        super(SASEBeam, self).__init__(**kargs)
        self.mu = mu
        self.sigma = sigma
        self.n_spikes = n_spikes
        
    def get_highest_wavenumber_beam(self):
        """
        For variable/polychromatic beam to return highest wavenumber.
        """
        return Beam(
            wavenumber=self.wavenumber*1.5,
            focus_x=self._focus_xFWHM,
            focus_y=self._focus_yFWHM,
            focus_shape=self._focus_shape,
            fluence=self.get_photons_per_pulse()
        )
    
    def generate_new_state(self):
        """
        For variable beam to return specific instance.
        """
        # If simple Beam, return itself.
        # Variable beams should return simple one.
        samples = np.random.normal(self.mu, self.sigma, self.n_spikes*100000)
        
        gkde = stats.gaussian_kde(samples)

        gkde.set_bandwidth(bw_method=0.25)

        xs = np.linspace(self.mu-50, self.mu+50, self.n_spikes+1)

        density, bins, patches = plt.hist(samples, bins=xs, histtype=u'step', density=True)
        
        ind = np.where(density == np.amax(density))
        density[ind[0][0]] *= 1.5
        density_renorm = density / density.sum()
        
        photon_energy = np.linspace(self.mu-50, self.mu+50, self.n_spikes+1).tolist()
        fluences = (self.get_photons_per_pulse()*density_renorm/density_renorm.sum())
        
        return [
            Beam(
                photon_energy=photon_energy[i],
                focus_x=self._focus_xFWHM,
                focus_y=self._focus_yFWHM,
                focus_shape=self._focus_shape,
                fluence=fluences[i])
            for i in range(self.n_spikes)
        ]
