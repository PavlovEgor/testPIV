import numpy as np
import matplotlib.pyplot as plt
import noise
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata


k = 512
n = 1024

dx = 1/k
tau = 1e-4

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

def generateInitDist(n=1024, numPoint=n * 10):

    sizeOfPoint = 2
    points = np.random.randint(sizeOfPoint, n-sizeOfPoint, size=(numPoint, 2))

    dist = np.zeros((n, n))

    for i in range(numPoint):
        for l1 in range(-sizeOfPoint, sizeOfPoint):
            for l2 in range(-sizeOfPoint, sizeOfPoint):
                dist[points[i][0]+ l1, points[i][1]+ l2] = 1.0

    return dist 

def culculeteVelocity(psi, k):
    Ux = np.zeros((k, k))
    Uy = np.zeros((k, k))

    n = psi.shape[0]
    for i in range(k):
        for j in range(k):
            Ux[i, j] = (psi[n//2 - k//2 + i,   n//2 - k//2 + j + 1] - psi[n//2 - k//2 +i,     n//2 - k//2 + j -1]) / (2 * dx)
            Uy[i, j] = -(psi[n//2 - k//2 + i+1, n//2 - k//2 + j] -     psi[n//2 - k//2 +i - 1, n//2 - k//2 + j]) / (2 * dx)

    return (Ux, Uy)

def culculeteNextDist(psi, initDist, tau, k):
    Ux, Uy = culculeteVelocity(psi, k)

    nextDist = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            nextDist[i, j] = initDist[n//2 - k//2 + i, n//2 - k//2 + j] - (tau / dx) * (Ux[i, j] * (initDist[n//2 - k//2 + i+1, n//2 - k//2 + j] - initDist[n//2 - k//2 + i-1, n//2 - k//2 + j]) + 
                                                            Uy[i, j] * (initDist[n//2 - k//2 + i, n//2 - k//2 + j + 1] - initDist[n//2 - k//2 + i, n//2 - k//2 + j - 1]))

    return nextDist

def move(psi, initDist, tau, k):
    Uy, Ux = culculeteVelocity(psi, k)

    nextDist = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            nextDist[i, j] = initDist[i, j] - (tau / dx) * (Ux[i, j] * (initDist[i+1, j] - initDist[i-1, j]) + 
                                                            Uy[i, j] * (initDist[i, j+1] - initDist[i, j-1]))

    return nextDist

psiFunc = lambda x, y: noise.pnoise2(
                                    x/100.0,
                                    y/100.0,
                                    octaves=100.0,
                                    persistence=0.1)

sizeOfPoint = 2
numPoint = 2 * n
points = np.random.randint(sizeOfPoint, n-sizeOfPoint, size=(numPoint, 2))

psi = generatePsi(n=1024, scale=100.0, octaves=2, persistence=0.1)
rho0 = generateInitDist(n=n, numPoint=2 * n)

plt.imshow(rho0, cmap='gray')
plt.colorbar()
plt.show()

rho1 = culculeteNextDist(psi, rho0, tau, n-1)

plt.imshow(rho1, cmap='gray')
plt.colorbar()
plt.show()

# plt.figure(figsize=(10, 8))
# contour = plt.contour(psi, levels=20, cmap='viridis')
# plt.colorbar(contour)
# plt.title("Контурный график перлинового шума")
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# x = np.linspace(0, 1, k)
# y = np.linspace(0, 1, k)
# X, Y = np.meshgrid(x, y)
# # График для Ux
# contour_psi = axes[0].contour(psi[n//2 - k//2: n//2 + k//2, n//2 - k//2: n//2 + k//2], levels=20, cmap='viridis')
# axes[0].set_title("psi")
# fig.colorbar(contour_psi, ax=axes[0])

# # График для Ux

# x = np.linspace(0, 1, k//16)
# y = np.linspace(0, 1, k//16)
# X, Y = np.meshgrid(x, y)
# Ux, Uy = culculeteVelocity(psi, k)

# axes[1].quiver(X, Y, Ux[::16, ::16], Uy[::16, ::16], color='blue', scale=500)
# axes[1].set_title("Векторное поле скорости (Ux, Uy")


# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots()

ax.contour(psi[n//2 - k//2: n//2 + k//2, n//2 - k//2: n//2 + k//2], levels=20, cmap='viridis')

x = np.linspace(0, k, k//16)
y = np.linspace(0, k, k//16)
X, Y = np.meshgrid(x, y)
Ux, Uy = culculeteVelocity(psi, k)

ax.quiver(X, Y, Uy[::16, ::16], Ux[::16, ::16], color='blue', scale=100)

plt.show()


fig, axes = plt.subplots(1, 4, figsize=(15, 5))

rho1 = culculeteNextDist(psi, rho0, tau, k)

x = np.linspace(0, 1, k)
y = np.linspace(0, 1, k)
X, Y = np.meshgrid(x, y)
# График для Ux
contour_psi = axes[0].contour(psi[n//2 - k//2: n//2 + k//2, n//2 - k//2: n//2 + k//2], levels=20, cmap='viridis')
axes[0].set_title("psi")
fig.colorbar(contour_psi, ax=axes[0])

# График для Ux
contour_Ux = axes[1].contourf(X, Y, rho0[n//2 - k//2: n//2 + k//2, n//2 - k//2: n//2 + k//2], levels=20, cmap='viridis')
axes[1].set_title("rho0")
fig.colorbar(contour_Ux, ax=axes[1])

# График для Uy
contour_Uy = axes[2].contourf(X, Y, rho1, levels=20, cmap='viridis')
axes[2].set_title("rho1")
fig.colorbar(contour_Uy, ax=axes[2])

contour_4 = axes[3].contourf(X, Y, rho1-rho0[n//2 - k//2: n//2 + k//2, n//2 - k//2: n//2 + k//2], levels=20, cmap='viridis')
axes[2].set_title("rho1-rho0")
fig.colorbar(contour_4, ax=axes[3])

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, rho0[n//2 - k//2: n//2 + k//2, n//2 - k//2: n//2 + k//2], levels=20, cmap='viridis')
fig.colorbar(contour)

T = np.linspace(0, 50/k, 50)

# Функция обновления поля для анимации
def update(frame):
    # Обновление поля A (например, добавление случайного шума)
    # rho1 = move(psi, rho1, T[frame], n-2*frame-4)
    rho1 = culculeteNextDist(psi, rho0, T[frame]/50, k)
    ax.clear()
    ax.set_title(f"Скалярное поле A (кадр {frame})")
    contour = ax.contourf(X, Y, rho1, levels=20, cmap='viridis')
    return contour

# Создание анимации
ani = FuncAnimation(fig, update, frames=50, interval=50, blit=False)

# Показать анимацию
plt.show()