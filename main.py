import Flow
import ModelPIV
import Particles

if __name__ == "__main__":
    numOfPixels = 1024
    dt = 10.0
    X_scale = 100 # mm
    Y_scale = 100 # mm


    flow = Flow.PenningFlow()
    particles = Particles.Particles(10 * numOfPixels, X_scale, Y_scale)
    model = ModelPIV.torchPIVModel(numOfPixels, particles)


    particles.evolve(flow, dt)
    model.predict(particles)
    err = model.error(flow)

    print("Error of model = ", err)