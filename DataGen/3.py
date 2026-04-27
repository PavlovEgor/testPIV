import numpy as np
import matplotlib.pyplot as plt
import noise
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from matplotlib.animation import PillowWriter


k = 512
n = 1024
numPoint=n * 2
sizeOfPoint = 2


dx = 1/k
tau = 10

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

def point2dist(points):
    dist = np.zeros((n, n))

    for i in range(numPoint):
        
        for l1 in range(-sizeOfPoint, sizeOfPoint):
            for l2 in range(-sizeOfPoint, sizeOfPoint):
                try:
                    dist[points[i][0]+ l1, points[i][1]+ l2] = 1.0
                except:
                    pass

    return dist 

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

def culculeteNextPoints(psi, points, tau):
    Ux, Uy = culculeteVelocity(psi)

    nextPoints = np.zeros_like(points)

    for i in range(numPoint):
        nextPoints[i] = points[i] + np.array([Ux[points[i][0], points[i][1]], Uy[points[i][0], points[i][1]]]) * tau
        
    return nextPoints 

def move(psi, initDist, tau, k):
    Uy, Ux = culculeteVelocity(psi, k)

    nextDist = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            nextDist[i, j] = initDist[i, j] - (tau / dx) * (Ux[i, j] * (initDist[i+1, j] - initDist[i-1, j]) + 
                                                            Uy[i, j] * (initDist[i, j+1] - initDist[i, j-1]))

    return nextDist



psi = generatePsi(n=1024, scale=500.0, octaves=2, persistence=0.1)

points = np.random.randint(sizeOfPoint, n-sizeOfPoint, size=(numPoint, 2))
nextPoints = culculeteNextPoints(psi, points, tau)

rho0 = point2dist(points)
rho1 = point2dist(nextPoints)


fig, axes = plt.subplots(1, 4, figsize=(15, 5))

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
# График для Ux
contour_psi = axes[0].contour(psi, levels=20, cmap='viridis')
axes[0].set_title("psi")
fig.colorbar(contour_psi, ax=axes[0])

# График для Ux
contour_Ux = axes[1].contourf(X, Y, rho0, levels=20, cmap='gray')
axes[1].set_title("rho0")
fig.colorbar(contour_Ux, ax=axes[1])

# График для Uy
contour_Uy = axes[2].contourf(X, Y, rho1, levels=20, cmap='gray')
axes[2].set_title("rho1")
fig.colorbar(contour_Uy, ax=axes[2])

contour_4 = axes[3].contourf(X, Y, rho1-rho0, levels=20, cmap='gray')
axes[2].set_title("rho1-rho0")
fig.colorbar(contour_4, ax=axes[3])

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, rho0, levels=20, cmap='gray')
fig.colorbar(contour)

T = np.linspace(0, tau, 50)

# Функция обновления поля для анимации
def update(frame):
    # Обновление поля A (например, добавление случайного шума)
    # rho1 = move(psi, rho1, T[frame], n-2*frame-4)
    nextPoints = culculeteNextPoints(psi, points, T[frame])
    rho1 = point2dist(nextPoints)
    ax.clear()
    ax.set_title(f"Скалярное поле A (кадр {frame})")
    contour = ax.contourf(X, Y, rho1, levels=20, cmap='gray')
    return contour

# Создание анимации
ani = FuncAnimation(fig, update, frames=50, interval=50, blit=False)

# Показать анимацию
ani.save("animation1.gif", writer=PillowWriter(fps=10))
plt.show()