from Particles.BasicParticles import BasicParticles

import numpy as np
from scipy.stats import uniform

class RandomUniformParticles(BasicParticles):

    def __init__(self, numOfParticles, X_scale, Y_scale, seed=42):

        super().__init__(numOfParticles, X_scale, Y_scale)

        self.deltaX = 0.05 * max(X_scale, Y_scale)

        self.x_dist = uniform(loc=-self.deltaX, scale=X_scale + self.deltaX)
        self.y_dist = uniform(loc=-self.deltaX, scale=Y_scale + self.deltaX)

        np.random.seed(seed)

        self.particles_coord_initial = np.array([self.x_dist.rvs(size=numOfParticles),
                                                self.y_dist.rvs(size=numOfParticles)])
        self.particles_coord_final = np.zeros_like(self.particles_coord_initial)

    def reset(self, newNumOfParticles):
        self.isEvolve = False
        self.numOfParticles = newNumOfParticles
        self.particles_coord_initial = np.array([self.x_dist.rvs(size=newNumOfParticles),
                                                self.y_dist.rvs(size=newNumOfParticles)])
        self.particles_coord_final = np.zeros_like(self.particles_coord_initial)