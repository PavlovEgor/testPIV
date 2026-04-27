import  numpy as np
import matplotlib.pyplot as plt

class BasicParticles:
    def __init__(self, numOfParticles, X_scale, Y_scale):
        self.isEvolve = False
        self.dt = None
        self.numOfParticles = numOfParticles
        self.X_scale = X_scale
        self.Y_scale = Y_scale

        self.particles_coord_initial = None 
        self.particles_coord_final = None 
        

    def reset(self, newNumOfParticles):
        raise NotImplementedError("Метод reset должен быть переопределен в дочернем классе")

    def evolve(self, flow, dt):

        self.particles_coord_final = self.particles_coord_initial + flow.velocity(self.particles_coord_initial[0],
                                                                                  self.particles_coord_initial[1]) * dt

        self.dt = dt
        self.isEvolve = True

    def plot_initial(self):
        if not self.isEvolve:
            raise ValueError("Сначала вызовите evolve()")

        plt.figure(figsize=(8, 8))
        plt.scatter(self.particles_coord_initial[0, :],
                    self.particles_coord_initial[1, :],
                    s=1, alpha=0.5)
        plt.xlim(0, self.X_scale)
        plt.ylim(0, self.Y_scale)
        plt.title("Initial particles distribution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_final(self):
        if not self.isEvolve:
            raise ValueError("Сначала вызовите evolve()")

        plt.figure(figsize=(8, 8))
        plt.scatter(self.particles_coord_final[0, :],
                    self.particles_coord_final[1, :],
                    s=1, alpha=0.5, c='red')
        plt.xlim(0, self.X_scale)
        plt.ylim(0, self.Y_scale)
        plt.title("Particles distribution after evolution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":

    import Flow

    numOfParticles = 2000
    dt = 1e-1

    flow = Flow.PenningFlow()
    particles = BasicParticles(numOfParticles)

    particles.plot_init_dist()
    particles.evolve(flow, dt)
    particles.plot_next_dist()

