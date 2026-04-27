from BasicFlow import BasicFlow

import opensimplex
import numpy as np

class OpensimplexFlow(BasicFlow):

    def __init__(self, scale=50.0, octaves=3, persistence=0.5, lacunarity=2.0, h=1e-5):

        super().__init__()
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.h = h


    def psi(self, X, Y):
        return opensimplex.noise2array(X, Y)

    def _gradient(self, X, Y):
        # U = dψ/dy, V = -dψ/dx
        psi_x_plus = self.psi(X + self.h, Y)
        psi_x_minus = self.psi(X - self.h, Y)
        dpsi_dx = (psi_x_plus - psi_x_minus) / (2 * self.h)

        psi_y_plus = self.psi(X, Y + self.h)
        psi_y_minus = self.psi(X, Y - self.h)
        dpsi_dy = (psi_y_plus - psi_y_minus) / (2 * self.h)

        U = dpsi_dy  # dx/dt = ∂ψ/∂y
        V = -dpsi_dx  # dy/dt = -∂ψ/∂x

        return U, V


if __name__ == "__main__":

    pass



