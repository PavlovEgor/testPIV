from Flow.BasicFlow import BasicFlow

import numpy as np

class DirectionalFlow(BasicFlow):

    def __init__(self, velocity):

        super().__init__()
        self.velocityVec = velocity

    def _gradient(self, X, Y):
        return self.velocityVec[1] * np.ones_like(X), self.velocityVec[0] * np.ones_like(X)


