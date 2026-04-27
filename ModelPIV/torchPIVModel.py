from ModelPIV.BasicModelPIV import BasicModelPIV

from torchPIV import OfflinePIV

import matplotlib.pyplot as plt
import numpy as np



class torchPIVModel(BasicModelPIV):

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
            initial_picture = self.generatePicture(condition="initial")
            final_picture = self.generatePicture(condition="final")

            self.savePicture(initial_picture, 'a.jpg', self.tmp_folder_name)
            self.savePicture(final_picture, 'b.jpg', self.tmp_folder_name)

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

        else:
            print("particles is not evolved yet")
            exit()


    def error(self, flow):
        VxGround, VyGround = flow.velocity(self.X, self.Y)

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


