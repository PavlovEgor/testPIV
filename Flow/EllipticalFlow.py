from Flow.BasicFlow import BasicFlow

import numpy as np

class EllipticalFlow(BasicFlow):

    def __init__(self, center, omega, e=0):

        super().__init__()

        self.center = center
        self.omega = omega

        self.e = e

        if e >= 1:
            print("incorrect Eccentricity for ellipse: 0 < e < 1")
            exit(1)


    def _gradient(self, X, Y):
        x_rel = X - self.center[0]
        y_rel = Y - self.center[1]
        r = np.sqrt(x_rel ** 2 + y_rel ** 2)

        with np.errstate(divide='ignore', invalid='ignore'):

            mask = r > 0
            U = np.zeros_like(X)
            V = np.zeros_like(Y)

            U[mask] = -self.omega * y_rel[mask]
            V[mask] = self.omega * x_rel[mask] * (np.sqrt(1 - self.e**2))

        return -V, U


