import numpy as np
import matplotlib.pyplot as plt

# Размер сетки
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

# Генерация случайных полей (замените на ваши данные)
Ux = np.random.rand(n, n)
Uy = np.random.rand(n, n)

# Создание фигуры с двумя графиками рядом
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# График для Ux
contour_Ux = axes[0].contourf(X, Y, Ux, levels=20, cmap='viridis')
axes[0].set_title("Поле Ux")
fig.colorbar(contour_Ux, ax=axes[0])

# График для Uy
contour_Uy = axes[1].contourf(X, Y, Uy, levels=20, cmap='plasma')
axes[1].set_title("Поле Uy")
fig.colorbar(contour_Uy, ax=axes[1])

plt.tight_layout()
plt.show()
