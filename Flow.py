import noise
import  numpy as np


class Flow:
    """Базовый класс для всех типов течений"""

    def __init__(self):
        pass

    def psi(self, x: np.float64, y: np.float64) -> np.float64:
        """Функция тока в точке (x, y). Должна быть переопределена"""
        raise NotImplementedError("Метод psi должен быть переопределен в дочернем классе")

    def psi_vectorized(self, X: np.array, Y: np.array) -> np.array:
        """Функция тока в точке (x, y). Должна быть переопределена"""
        raise NotImplementedError("Метод psi_vectorized должен быть переопределен в дочернем классе")

    def _gradient(self, X: np.array, Y: np.array) -> (np.array, np.array):
        """Функция тока в точке (x, y). Должна быть переопределена"""
        raise NotImplementedError("Метод _gradient должен быть переопределен в дочернем классе")

    def get_velocity_matrices(self, n):
        """Построение матриц скоростей U и V на сетке n x n"""
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)

        U, V = self._gradient(X, Y)
        return U, V, X, Y

    def velocity(self, x, y):
        """Скорость в конкретной точке"""
        X = np.array([[x]])
        Y = np.array([[y]])
        U, V = self._gradient(X, Y)
        return np.array([U[0, 0], V[0, 0]])

    def velocity_vectorized(self, points):
        """
        points: массив shape (N, 2) с координатами x, y
        возвращает: массивы U, V shape (N,)
        """
        points = np.asarray(points)
        X = points[0, :].reshape(-1, 1)
        Y = points[1, :].reshape(-1, 1)
        U, V = self._gradient(X, Y)
        return np.array([U.flatten(), V.flatten()])


class PenningFlow(Flow):

    def __init__(self, scale=50.0, octaves=3, persistence=0.5, lacunarity=2.0, h=1e-5):

        super().__init__()
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.h = h

    def psi(self, x, y):
        return noise.pnoise2(
            x / self.scale,
            y / self.scale,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
        )

    def psi_vectorized(self, X, Y):
        """Векторизованная версия для матриц"""
        shape = X.shape
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        result = np.zeros_like(X_flat)
        for i, (x, y) in enumerate(zip(X_flat, Y_flat)):
            result[i] = self.psi(x, y)

        return result.reshape(shape)

    def _gradient(self, X, Y):
        """Численный градиент"""
        psi_val = self.psi_vectorized(X, Y)

        # U = dψ/dy, V = -dψ/dx
        psi_x_plus = self.psi_vectorized(X + self.h, Y)
        psi_x_minus = self.psi_vectorized(X - self.h, Y)
        dpsi_dx = (psi_x_plus - psi_x_minus) / (2 * self.h)

        psi_y_plus = self.psi_vectorized(X, Y + self.h)
        psi_y_minus = self.psi_vectorized(X, Y - self.h)
        dpsi_dy = (psi_y_plus - psi_y_minus) / (2 * self.h)

        U = dpsi_dy  # dx/dt = ∂ψ/∂y
        V = -dpsi_dx  # dy/dt = -∂ψ/∂x

        return U, V