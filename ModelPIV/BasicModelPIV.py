import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import os


class BasicModelPIV:

    def __init__(self, numOfPixelsX, particles):
        self.numOfPixelsX = numOfPixelsX
        self.numOfPixelsY = int(self.numOfPixelsX * particles.Y_scale / particles.X_scale)
        self.particles = particles

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

    def predict(self, particles):
        raise NotImplementedError("Метод predict должен быть переопределен")

    def error(self, flow):
        raise NotImplementedError("Метод error должен быть переопределен")

