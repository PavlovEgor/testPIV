import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import os


class BasicModelPIV:

    def __init__(self, numOfPixelsX, particles):
        self.numOfPixelsX = numOfPixelsX
        self.numOfPixelsY = int(self.numOfPixelsX * particles.Y_scale / particles.X_scale)
        self.particles = particles

        self.X = None
        self.Y = None
        self.Vx = None
        self.Vy = None

        self.VxGround = None
        self.VyGround = None

    def generatePicture(self, condition="initial"):

        picture = np.zeros((self.numOfPixelsX, self.numOfPixelsY))

        tmp_prod_array = np.array([(self.numOfPixelsX - 1)/self.particles.X_scale,
                                    (self.numOfPixelsY - 1)/self.particles.Y_scale])

        if condition=="initial":
            pixels = (self.particles.particles_coord_initial.T * tmp_prod_array).astype(int).T
        else:
            pixels = (self.particles.particles_coord_final.T * tmp_prod_array).astype(int).T


        valid_pixels = (pixels[0, :] >= 0) & (pixels[0, :] < self.numOfPixelsX) & \
                       (pixels[1, :] >= 0) & (pixels[1, :] < self.numOfPixelsY)

        pixels = pixels[:, valid_pixels]
        picture[pixels[0, :], pixels[1, :]] = 255


        picture = gaussian_filter(picture, sigma=0.5)

        return picture.T

    def savePicture(self, picture, filename="particles.png", folder="."):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        picture = (picture / picture.max() * 255).astype(np.uint8)

        img = Image.fromarray(picture)
        img.save(filepath)
        print(f"Изображение сохранено как {filepath}")

    def plot_velocity(self, ax):

        step = max(1, self.Vx.shape[0] // 20)  # разреживаем стрелки для читаемости
        x_plot = self.X[::step, ::step]
        y_plot = self.Y[::step, ::step]
        vx_plot1 = self.Vx[::step, ::step]
        vy_plot1 = self.Vy[::step, ::step]
        vx_plot2 = self.VxGround[::step, ::step]
        vy_plot2 = self.VyGround[::step, ::step]

        ax.quiver(x_plot, y_plot, vx_plot1, vy_plot1, alpha=0.8)
        ax.quiver(x_plot, y_plot, vx_plot2, vy_plot2, alpha=0.8, color='red')

        ax.set_title('Model (self.Vx, self.Vy)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def predict(self, particles):
        raise NotImplementedError("Метод predict должен быть переопределен")

    def error(self, flow):
        raise NotImplementedError("Метод error должен быть переопределен")
