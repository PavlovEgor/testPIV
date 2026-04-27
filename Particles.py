import  numpy as np
import matplotlib.pyplot as plt

class Particles:
    def __init__(self, numOfParticles, X_scale, Y_scale, seed=42):
        self.isEvolve = False
        self.dt = None
        self.numOfParticles = numOfParticles
        self.X_scale = X_scale
        self.Y_scale = Y_scale

        self.deltaX = 0.05 * max(X_scale, Y_scale)

        self.particles_coord_old = np.array([np.random.uniform(low=0-self.deltaX, high=X_scale+self.deltaX, size=numOfParticles),
                                             np.random.uniform(low=0-self.deltaX, high=Y_scale+self.deltaX, size=numOfParticles)])
        self.particles_coord_new = np.zeros_like(self.particles_coord_old)

    def set_numOfParticles(self, newNumOfParticles):
        self.isEvolve = False
        self.numOfParticles = newNumOfParticles
        self.particles_coord_old = np.random.uniform(low=0, high=1, size=(2, newNumOfParticles))
        self.particles_coord_new = np.zeros_like(self.particles_coord_old)

    def evolve(self, flow, dt):

        self.particles_coord_new = self.particles_coord_old + flow.velocity_vectorized(self.particles_coord_old) * dt

        print("mean diff:", np.std(self.particles_coord_new - self.particles_coord_old))

        self.dt = dt
        self.isEvolve = True

    def plot_init_dist(self):
        """Визуализация начального распределения частиц"""
        plt.figure(figsize=(8, 8))
        plt.scatter(self.particles_coord_old[0, :],
                    self.particles_coord_old[1, :],
                    s=1, alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Initial particles distribution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_next_dist(self):
        """Визуализация конечного распределения частиц"""
        if not self.isEvolve:
            raise ValueError("Сначала вызовите evolve()")

        plt.figure(figsize=(8, 8))
        plt.scatter(self.particles_coord_new[0, :],
                    self.particles_coord_new[1, :],
                    s=1, alpha=0.5, c='red')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
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
    particles = Particles(numOfParticles)

    particles.plot_init_dist()
    particles.evolve(flow, dt)
    particles.plot_next_dist()

