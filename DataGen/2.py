import numpy as np
import matplotlib.pyplot as plt
import noise
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from matplotlib.animation import PillowWriter


k = 100
n = 256

dx = 1/k
tau = 5e-3

def generatePsi(n=1024, scale=100.0, octaves=2, persistence=0.1):
    
    psi = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            psi[i][j] = noise.pnoise2(
                                    i/scale,
                                    j/scale,
                                    octaves=octaves,
                                    persistence=persistence)
            
    return psi

def generateInitDist():
    pass 

def culculeteVelocity(psi):
    global n

    Ux = np.zeros((n, n))
    Uy = np.zeros((n, n))

    Ux[0, :] = (psi[1, :] - psi[0, :]) / dx
    Ux[-1, :] = (psi[-1, :] - psi[-2, :]) / dx
    Uy[:, 0] = (psi[:, 1] - psi[:, 0]) / dx
    Uy[:, -1] = (psi[:, -1] - psi[:, -2]) / dx

    for i in range(1, n-1):
        for j in range(1, n-1):
            Ux[i, j] = (psi[i+1, j] - psi[i-1, j]) / (2 * dx)
            Uy[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2 * dx)

    return (Ux, Uy)


psi = generatePsi(n=n, scale=100.0, octaves=2, persistence=0.1)
rho0 = generatePsi(n=n, scale=10, octaves=2, persistence=0.1)



x0 = np.linspace(0, 1, n)
y0 = np.linspace(0, 1, n)

X, Y = np.meshgrid(x0, y0)
points0 = np.vstack((X.ravel(), Y.ravel())).T

Ux, Uy = culculeteVelocity(psi)

fig, ax = plt.subplots()

ax.contour(psi, levels=20, cmap='viridis')

plt.show()

Ux_flat = Ux.ravel()
Uy_flat = Uy.ravel()
rho_flat = rho0.ravel()

pointsTau = np.zeros_like(points0)
pointsTau.T[0] = points0.T[0] + tau * Ux_flat
pointsTau.T[1] = points0.T[1] + tau * Uy_flat


# Новая равномерная сетка
xi = np.linspace(0.2, 0.8, k)  # координаты x для новой сетки
yi = np.linspace(0.2, 0.8, k)  # координаты y для новой сетки
xi_grid, yi_grid = np.meshgrid(xi, yi)

interpolated0 = griddata(points0, rho_flat, (xi_grid, yi_grid), method='cubic')
interpolatedTau = griddata(pointsTau, rho_flat, (xi_grid, yi_grid), method='cubic')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# График для Ux
contour_1 = axes[0].contourf(xi_grid, yi_grid, interpolated0.reshape(k, k), levels=20, cmap='viridis')
axes[0].set_title("rho0")
fig.colorbar(contour_1, ax=axes[0])

# График для Ux
contour_Ux = axes[1].contourf(xi_grid, yi_grid, interpolatedTau.reshape(k, k), levels=20, cmap='viridis')
axes[1].set_title("rho1")
fig.colorbar(contour_Ux, ax=axes[1])


plt.tight_layout()
plt.show()

plt.contourf(xi_grid, yi_grid, interpolated0.reshape(k, k), levels=20, cmap='viridis')
plt.savefig("rho0.jpg")

plt.contourf(xi_grid, yi_grid, interpolatedTau.reshape(k, k), levels=20, cmap='viridis')
plt.savefig("rho1.jpg")

# fig, ax = plt.subplots(figsize=(8, 6))
# contour = ax.contourf(xi_grid, yi_grid, interpolated0.reshape(k, k), levels=20, cmap='viridis')
# fig.colorbar(contour)

# T = np.linspace(0, 5/k, 50)

# # Функция обновления поля для анимации
# def update(frame):
#     # Обновление поля A (например, добавление случайного шума)
#     pointsTau.T[0] = points0.T[0] + (T[frame]) * Ux_flat
#     pointsTau.T[1] = points0.T[1] + (T[frame]) * Uy_flat

#     interpolatedTau = griddata(pointsTau, rho_flat, (xi_grid, yi_grid), method='cubic')

#     ax.clear()
#     ax.set_title(f"Скалярное поле A (кадр {frame})")
#     contour = ax.contourf(xi_grid, yi_grid, interpolatedTau.reshape(k, k), levels=20, cmap='viridis')
#     return contour

# # Создание анимации
# ani = FuncAnimation(fig, update, frames=10, interval=10, blit=False)

# ani.save("animation.gif", writer=PillowWriter(fps=10))
# # Показать анимацию
# plt.show()