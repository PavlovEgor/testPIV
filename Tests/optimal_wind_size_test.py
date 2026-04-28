import numpy as np

from Flow.PenningFlow import PenningFlow
from ModelPIV.torchPIVModel import torchPIVModel
from Particles.RandomUniformParticles import RandomUniformParticles

import matplotlib.pyplot as plt

def optimal_wind_size_test(flow, particles, model):

    particles.evolve(flow, dt)

    wind_sizes = np.linspace(16, 256, 10).astype(int)
    errs = np.zeros_like(wind_sizes, dtype=np.float64)
    for i, ws in enumerate(wind_sizes):
        model.set_setting(wind_size=ws, overlap=ws//2)
        model.predict(particles)
        errs[i] = model.error(flow)

    plt.plot(wind_sizes, errs, 'o-')

    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    numOfPixels = 1024
    dt = 10.0
    X_scale = 100 # mm
    Y_scale = 100 # mm

    flow = PenningFlow()
    particles = RandomUniformParticles(10 * numOfPixels, X_scale, Y_scale)
    model = torchPIVModel(numOfPixels, particles)

    optimal_wind_size_test(flow, particles, model)

