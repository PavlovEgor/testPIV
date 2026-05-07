import numpy as np

from Flow.PenningFlow import PenningFlow
from ModelPIV.torchPIVModel import torchPIVModel
from Particles.RandomUniformParticles import RandomUniformParticles

import matplotlib.pyplot as plt

def optimal_numOfParticles_test(flow, particles, model):


    numOfParticles = np.linspace(1 * model.numOfPixelsX, 3500 * model.numOfPixelsX, 10).astype(int)
    errs1 = np.zeros_like(numOfParticles, dtype=np.float64)
    errs2 = np.zeros_like(numOfParticles, dtype=np.float64)

    for i, nop in enumerate(numOfParticles):

        particles.reset(nop)
        particles.evolve(flow, dt)
        model.predict(particles)

        errs1[i] = model.error(flow)

        model.correct()

        errs2[i] = model.error(flow)


    plt.plot(numOfParticles, errs1, 'o-', label="base")
    plt.plot(numOfParticles, errs2, 'o-', label="corrected")

    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    numOfPixels = 1024
    dt = 10.0
    X_scale = 100 # mm
    Y_scale = 100 # mm

    flow = PenningFlow()
    particles = RandomUniformParticles(10 * numOfPixels, X_scale, Y_scale)
    model = torchPIVModel(numOfPixels, particles)

    optimal_numOfParticles_test(flow, particles, model)

