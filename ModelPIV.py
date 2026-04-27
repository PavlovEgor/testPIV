import numpy as np
from torchPIV import OfflinePIV
from scipy.ndimage import gaussian_filter
from PIL import Image
import os
import matplotlib.pyplot as plt

import Particles
import Flow

class BasicModel:

    def __init__(self, numOfPixelsX, particles):
        self.numOfPixelsX = numOfPixelsX
        self.numOfPixelsY = int(self.numOfPixelsX * particles.Y_scale / particles.X_scale)
        self.particles = particles

    def _coord_to_pixel(self, coords):
        """Преобразование координат [0,1] в пиксельные координаты"""
        return (coords * (np.array([(self.numOfPixelsX - 1)/self.particles.X_scale,
                                    (self.numOfPixelsY - 1)/self.particles.Y_scale]))).astype(int)

    def generatePictures(self):
        """
        Генерация изображения по координатам частиц
        particles_coord: массив (N, 2) с координатами частиц
        возвращает: numpy array (numOfPixels, numOfPixels)
        """
        # Создаем черное изображение
        pictureA = np.zeros((self.numOfPixelsX, self.numOfPixelsY))
        pictureB = np.zeros_like(pictureA)

        # Конвертируем координаты в пиксели
        pixelsA = (self.particles.particles_coord_old.T *
                   (np.array([(self.numOfPixelsX - 1)/self.particles.X_scale,
                                    (self.numOfPixelsY - 1)/self.particles.Y_scale]))).astype(int).T

        pixelsB = (self.particles.particles_coord_new.T *
                   (np.array([(self.numOfPixelsX - 1)/self.particles.X_scale,
                                    (self.numOfPixelsY - 1)/self.particles.Y_scale]))).astype(int).T


        # Добавляем частицы (каждая частица - один пиксель)
        valid_pixelsA = (pixelsA[0, :] >= 0) & (pixelsA[0, :] < self.numOfPixelsX) & \
                       (pixelsA[1, :] >= 0) & (pixelsA[1, :] < self.numOfPixelsY)

        valid_pixelsB = (pixelsB[0, :] >= 0) & (pixelsB[0, :] < self.numOfPixelsX) & \
                       (pixelsB[1, :] >= 0) & (pixelsB[1, :] < self.numOfPixelsY)

        pixelsA = pixelsA[:, valid_pixelsA]
        pictureA[pixelsA[0, :], pixelsA[1, :]] = 255

        pixelsB = pixelsB[:, valid_pixelsB]
        pictureB[pixelsB[0, :], pixelsB[1, :]] = 255

        # Опционально: добавляем сглаживание (размытие)
        pictureA = gaussian_filter(pictureA, sigma=0.5)

        pictureB = gaussian_filter(pictureB, sigma=0.5)

        return pictureA.T, pictureB.T

    def savePicture(self, picture, filename="particles.png", folder="."):
        """Сохранение изображения"""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        # Нормализуем изображение
        picture = (picture / picture.max() * 255).astype(np.uint8)

        img = Image.fromarray(picture)
        img.save(filepath)
        print(f"Изображение сохранено как {filepath}")

    def predict(self, particles):
        """Прогнозирование поля скорости - должно быть переопределено"""
        raise NotImplementedError("Метод predict должен быть переопределен")

    def error(self, flow):
        """Вычисление ошибки - должно быть переопределено"""
        raise NotImplementedError("Метод error должен быть переопределен")


class torchPIVModel(BasicModel):

    def __init__(self, numOfPixelsX, numOfPixelsY):
        super().__init__(numOfPixelsX, numOfPixelsY)

        self.tmp_folder_name = "tmp"
        self.folder_mode = "pairs"
        self.file_fmt = "jpg"

        self.device = "cpu"

        self.wind_size = 128
        self.overlap = 32
        self.multipass = 2
        self.multipass_mode = "DWS"
        self.multipass_scale = 2.0

        self.X = None
        self.Y = None
        self.Vx = None
        self.Vy = None

    def predict(self, particles):

        if particles.isEvolve:
            A, B = self.generatePictures()

            self.savePicture(A, 'a.jpg', self.tmp_folder_name)
            self.savePicture(B, 'b.jpg', self.tmp_folder_name)

            piv_gen = OfflinePIV(
                folder=self.tmp_folder_name,  # Path to experiment
                device=self.device,  # Device name
                file_fmt=self.file_fmt,
                wind_size=self.wind_size,
                overlap=self.overlap,
                dt=particles.dt,  # Time between frames, mcs
                scale=particles.X_scale/self.numOfPixelsX,  # mm/pix
                multipass=self.multipass,
                multipass_mode=self.multipass_mode,  # CWS or DWS
                multipass_scale=self.multipass_scale,  # Window downscale on each pass
                folder_mode=self.folder_mode  # Pairs or sequential frames
            )

            results = []
            for out in piv_gen():
                results.append(out)
            self.X, self.Y, self.Vx, self.Vy = results[0]

            # self.X = self.particles.X_scale - self.X
            # self.Y = self.particles.Y_scale - self.Y

        else:
            print("particles is not evolved yet")
            exit()


    def error(self, flow):
        VxGround, VyGround = flow.velocity_vectorized(np.array([self.X.reshape(-1), self.Y.reshape(-1)]))

        VxGround = np.reshape(VxGround, self.Vx.shape)
        VyGround = np.reshape(VyGround, self.Vy.shape)

        VxGround = np.flipud(VxGround)
        VyGround = np.flipud(VyGround)

        VyGround *= -1

        # Рисуем два векторных поля
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Поле 1: исходное (self.Vx, self.Vy)
        step = max(1, self.Vx.shape[0] // 20)  # разреживаем стрелки для читаемости
        x_plot = self.X[::step, ::step]
        y_plot = self.Y[::step, ::step]
        vx_plot1 = self.Vx[::step, ::step]
        vy_plot1 = self.Vy[::step, ::step]
        vx_plot2 = VxGround[::step, ::step]
        vy_plot2 = VyGround[::step, ::step]

        ax1.quiver(x_plot, y_plot, vx_plot1, vy_plot1, alpha=0.8)
        ax1.set_title('Model (self.Vx, self.Vy)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        ax1.quiver(x_plot, y_plot, vx_plot2, vy_plot2, alpha=0.8, color='red')

        # Поле 2: вычисленное (VxGround, VyGround)
        ax2.quiver(x_plot, y_plot, vx_plot2, vy_plot2, alpha=0.8, color='red')
        ax2.set_title('Ground (VxGround, VyGround)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return np.sqrt(np.mean((self.Vx - VxGround) ** 2) + np.mean((self.Vy - VyGround) ** 2))


