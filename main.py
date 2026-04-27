from Flow.PenningFlow import PenningFlow
from ModelPIV.torchPIVModel import torchPIVModel
from Particles.RandomUniformParticles import RandomUniformParticles

if __name__ == "__main__":
    numOfPixels = 1024
    dt = 10.0
    X_scale = 100 # mm
    Y_scale = 100 # mm

    flow = PenningFlow()
    particles = RandomUniformParticles(10 * numOfPixels, X_scale, Y_scale)
    model = torchPIVModel(numOfPixels, particles)

    particles.evolve(flow, dt)
    model.predict(particles)
    err = model.error(flow)

    print("Error of model = ", err)