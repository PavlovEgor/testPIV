import  numpy as np


class BasicFlow:

    def __init__(self):
        pass

    def psi_at_point(self, x: np.float64, y: np.float64) -> np.float64:
        raise NotImplementedError("Метод psi должен быть переопределен в дочернем классе")

    def psi(self, X: np.array, Y: np.array) -> np.array:
        raise NotImplementedError("Метод psi_vectorized должен быть переопределен в дочернем классе")

    def _gradient(self, X: np.array, Y: np.array):
        raise NotImplementedError("Метод _gradient должен быть переопределен в дочернем классе")

    def velocity(self, X: np.array, Y: np.array) -> np.array:

        mV, U = self._gradient(X, Y)

        return np.array([U, -mV])
