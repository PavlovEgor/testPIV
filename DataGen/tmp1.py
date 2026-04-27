import numpy as np
import noise
import matplotlib.pyplot as plt


class Flow:
    def __init__(self, scale=100.0, octaves=2, persistence=0.5, lacunarity=2.0):
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

    def psi(self, x, y):
        """Функция тока"""
        return noise.pnoise2(
            x / self.scale,
            y / self.scale,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            repeatx=1024,  # для периодичности
            repeaty=1024,
            base=0
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

    def _gradient(self, X, Y, h=1e-5):
        """Численный градиент"""
        psi_val = self.psi_vectorized(X, Y)

        # U = dψ/dy, V = -dψ/dx
        psi_x_plus = self.psi_vectorized(X + h, Y)
        psi_x_minus = self.psi_vectorized(X - h, Y)
        dpsi_dx = (psi_x_plus - psi_x_minus) / (2 * h)

        psi_y_plus = self.psi_vectorized(X, Y + h)
        psi_y_minus = self.psi_vectorized(X, Y - h)
        dpsi_dy = (psi_y_plus - psi_y_minus) / (2 * h)

        U = dpsi_dy  # dx/dt = ∂ψ/∂y
        V = -dpsi_dx  # dy/dt = -∂ψ/∂x

        return U, V

    def get_velocity_matrices(self, n, x_range=(-100, 100), y_range=(-100, 100)):
        """Построение матриц скоростей U и V на сетке n x n"""
        x = np.linspace(x_range[0], x_range[1], n)
        y = np.linspace(y_range[0], y_range[1], n)
        X, Y = np.meshgrid(x, y)

        U, V = self._gradient(X, Y)
        return U, V, X, Y

    def velocity_at_point(self, x, y):
        """Скорость в конкретной точке"""
        X = np.array([[x]])
        Y = np.array([[y]])
        U, V = self._gradient(X, Y)
        return U[0, 0], V[0, 0]

    def velocity_at_points(self, points):
        """
        points: массив shape (N, 2) с координатами x, y
        возвращает: массивы U, V shape (N,)
        """
        points = np.asarray(points)
        X = points[:, 0].reshape(-1, 1)
        Y = points[:, 1].reshape(-1, 1)
        U, V = self._gradient(X, Y)
        return U.flatten(), V.flatten()

    def plot_psi(self, n, x_range=(-100, 100), y_range=(-100, 100), fig=None, ax=None, levels=20):
        """Отрисовка функции тока"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        x = np.linspace(x_range[0], x_range[1], n)
        y = np.linspace(y_range[0], y_range[1], n)
        X, Y = np.meshgrid(x, y)
        psi_vals = self.psi_vectorized(X, Y)

        contour = ax.contour(X, Y, psi_vals, levels=levels, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Stream function')
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax)

        return fig, ax

    def plot_streamlines(self, n, x_range=(-100, 100), y_range=(-100, 100),
                         fig=None, ax=None, levels=20, density=2.0):
        """Отрисовка линий тока"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        U, V, X, Y = self.get_velocity_matrices(n, x_range, y_range)

        # Линии тока через streamplot
        strm = ax.streamplot(X, Y, U, V, color='blue', density=density,
                             linewidth=1, arrowsize=1)

        # Наложим контуры функции тока
        psi_vals = self.psi_vectorized(X, Y)
        contour = ax.contour(X, Y, psi_vals, levels=levels,
                             cmap='viridis', alpha=0.6, linestyles='dashed')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Streamlines with velocity vectors')
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax)

        return fig, ax

    def plot_velocity_field(self, n, x_range=(-100, 100), y_range=(-100, 100),
                            fig=None, ax=None, density=2):
        """Отрисовка поля скоростей (стрелки)"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        U, V, X, Y = self.get_velocity_matrices(n, x_range, y_range)

        # Прореживаем векторы для читаемости
        step = max(1, n // density)
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  U[::step, ::step], V[::step, ::step],
                  alpha=0.7, width=0.003)

        # Добавим цветовую карту для величины скорости
        speed = np.sqrt(U ** 2 + V ** 2)
        im = ax.contourf(X, Y, speed, levels=20, cmap='plasma', alpha=0.3)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Velocity field with speed magnitude')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Speed')

        return fig, ax