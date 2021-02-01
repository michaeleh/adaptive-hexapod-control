import numpy as np

import nengo


def angle(inp):
    x, y = inp
    return np.arctan2(y, x)


class PlaneRotation:
    def __init__(self, sim_dt, dist):
        self.model = nengo.Network()
        self.values1 = [0]
        self.values2 = [0]
        self.dist = dist

        with self.model:
            stim1 = nengo.Node(lambda t: self.values1[-1])
            stim2 = nengo.Node(lambda t: self.values2[-1])
            # integrator 1
            pos1 = nengo.Ensemble(n_neurons=500, dimensions=1, radius=100)
            nengo.Connection(pos1, pos1, synapse=1e-2)
            nengo.Connection(stim1, pos1, transform=5, synapse=1e-2)
            # integrator 2
            pos2 = nengo.Ensemble(n_neurons=500, dimensions=1, radius=100)
            nengo.Connection(pos2, pos2, synapse=1e-2)
            nengo.Connection(stim2, pos2, transform=5, synapse=1e-2)

            # angle of the slop, [y diff (constant), y diff (height)]
            y_diff = nengo.Ensemble(n_neurons=500, dimensions=2, radius=200)

            d_stim = nengo.Node([self.dist])  # x distance is constant
            nengo.Connection(d_stim, y_diff[0])

            nengo.Connection(pos1, y_diff[1])
            nengo.Connection(pos2, y_diff[1], transform=-1)

            # find angle
            y_rot = nengo.Ensemble(n_neurons=500, dimensions=1, radius=np.pi / 2)
            nengo.Connection(y_diff, y_rot, synapse=0.01, function=angle)

            self.rot_probe = nengo.Probe(y_diff[1], synapse=1e-2)

        self.sim = nengo.Simulator(self.model, dt=sim_dt)

    def update(self, val1, val2):
        self.values1.append(val1)
        self.values2.append(val2)
        self.sim.step()

    def get_xy(self):
        return self.sim.trange(), self.sim.data[self.rot_probe]
