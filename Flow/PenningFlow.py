from Flow.BasicFlow import BasicFlow

import noise
import numpy as np

class PenningFlow(BasicFlow):

    def __init__(self, scale=50.0, octaves=3, persistence=0.5, lacunarity=2.0, h=1e-5):

        super().__init__()
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.h = h

    def psi_at_point(self, x, y):
        return noise.pnoise2(
            x / self.scale,
            y / self.scale,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
        )

    def psi(self, X, Y):
        shape = X.shape
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        result = np.zeros_like(X_flat)
        for i, (x, y) in enumerate(zip(X_flat, Y_flat)):
            result[i] = self.psi_at_point(x, y)

        return result.reshape(shape)

    def _gradient(self, X, Y):
        psi_x_plus = self.psi(X + self.h, Y)
        psi_x_minus = self.psi(X - self.h, Y)
        dpsi_dx = (psi_x_plus - psi_x_minus) / (2 * self.h)

        psi_y_plus = self.psi(X, Y + self.h)
        psi_y_minus = self.psi(X, Y - self.h)
        dpsi_dy = (psi_y_plus - psi_y_minus) / (2 * self.h)

        return dpsi_dx, dpsi_dy


if __name__ == "__main__":

    flow = PenningFlow()

