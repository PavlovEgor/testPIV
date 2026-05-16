from Flow.PenningFlow import PenningFlow
from Flow.EllipticalFlow import EllipticalFlow
from Flow.DirectionalFlow import DirectionalFlow
from ModelPIV.torchPIVModel import torchPIVModel
from Particles.RandomUniformParticles import RandomUniformParticles

import matplotlib.pyplot as plt
import numpy as np

def simple_test(flow, particles, model):

    particles.evolve(flow, dt)
    model.predict(particles)
    # model.correct()
    err = model.error(flow, 'L2', n=1)

    print("Error of model = ", err)

    fig, ax = plt.subplots()

    model.plot_velocity(ax)

    plt.show()

    model.correct()

    err = model.error(flow, 'L2', n=1)

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
    # flow = DirectionalFlow(velocity=np.array([0.2, 0.2]))
    # flow = EllipticalFlow(center=np.array([X_scale / 2, Y_scale / 2]), omega=0.01, e=0.999)

    particles = RandomUniformParticles(10 * numOfPixels, X_scale, Y_scale)
    model = torchPIVModel(numOfPixels, particles)

    simple_test(flow, particles, model)

