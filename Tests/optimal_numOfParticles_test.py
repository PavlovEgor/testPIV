import numpy as np

from Flow.PenningFlow import PenningFlow
from ModelPIV.torchPIVModel import torchPIVModel
from Particles.RandomUniformParticles import RandomUniformParticles

import matplotlib.pyplot as plt

def optimal_numOfParticles_test(flow, particles, model):


    numOfParticles = np.linspace(100 * model.numOfPixelsX, 500 * model.numOfPixelsX, 10).astype(int)
    errs = np.zeros_like(numOfParticles)
    for i, nop in enumerate(numOfParticles):

        particles.reset(nop)
        particles.evolve(flow, dt)
        model.predict(particles)

        errs[i] = model.error(flow)

    plt.plot(numOfParticles, errs, 'o-')

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

    optimal_numOfParticles_test(flow, particles, model)

