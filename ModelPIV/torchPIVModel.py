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

        self.wind_size = 64
        self.overlap = 32
        self.multipass = 2
        self.multipass_mode = "DWS"
        self.multipass_scale = 2.0

    def set_setting(self,
                    tmp_folder_name="tmp",
                    folder_mode="pairs",
                    file_fmt="jpg",
                    device="cpu",
                    wind_size=32,
                    overlap=16,
                    multipass=2,
                    multipass_mode="CWS",
                    multipass_scale=2.0
                    ):
        self.tmp_folder_name = tmp_folder_name
        self.folder_mode = folder_mode
        self.file_fmt = file_fmt

        self.device = device

        self.wind_size = wind_size
        self.overlap = overlap
        self.multipass = multipass
        self.multipass_mode = multipass_mode
        self.multipass_scale = multipass_scale

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
                dt=particles.dt*1000,  # Time between frames, mcs
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

    def error(self, flow, n=1):

        self.VxGround, self.VyGround = flow.velocity(self.X, self.Y)

        L = self.wind_size * self.particles.X_scale / self.numOfPixelsX

        for i in range(1, n+1):
            for j in range(1, n+1):
                VxGround_tmp, VyGround_tmp = flow.velocity(self.X - L/2 + j * L / (n+1), self.Y - L/2 + i * L / (n+1))

                self.VxGround += VxGround_tmp
                self.VyGround += VyGround_tmp

        self.VxGround /= n + 1
        self.VyGround /= n + 1

        self.VxGround = np.reshape(self.VxGround, self.Vx.shape)
        self.VyGround = np.reshape(self.VyGround, self.Vy.shape)

        self.VxGround = np.flipud(self.VxGround)
        self.VyGround = np.flipud(self.VyGround)

        self.VyGround *= -1

        return np.sqrt(np.mean((self.Vx - self.VxGround) ** 2) + np.mean((self.Vy - self.VyGround) ** 2))
