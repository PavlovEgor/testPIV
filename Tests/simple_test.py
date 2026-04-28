from Flow.PenningFlow import PenningFlow
from ModelPIV.torchPIVModel import torchPIVModel
from Particles.RandomUniformParticles import RandomUniformParticles

import matplotlib.pyplot as plt
import numpy as np

def simple_test(flow, particles, model):

    particles.evolve(flow, dt)
    model.predict(particles)
    err = model.error(flow, 1)

    print("Error of model = ", err)

    fig, ax = plt.subplots()

    model.plot_velocity(ax)

    plt.show()

if __name__ == "__main__":

    numOfPixels = 1024
    dt = 5.0 # ms
    X_scale = 100 # mm
    Y_scale = 100 # mm

    flow = PenningFlow()
    particles = RandomUniformParticles(10 * numOfPixels, X_scale, Y_scale)
    model = torchPIVModel(numOfPixels, particles)

    simple_test(flow, particles, model)

