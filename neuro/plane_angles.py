import pickle

import numpy as np

import nengo

from neuro.utils import SCALE


def angle(inp):
    x, y = inp
    return np.arctan2(y, x)


class PlaneRotation:
    def __init__(self, sim_dt, dist):
        self.model = nengo.Network()
        self.values1 = [0]
        self.values2 = [0]
        self.dist = dist
        self.sim_dt = sim_dt

        with self.model:
            stim1 = nengo.Node(lambda t: self.values1[-1])
            stim2 = nengo.Node(lambda t: self.values2[-1])

            # integrator 1
            pos1 = nengo.Ensemble(n_neurons=500, dimensions=1, radius=SCALE)
            nengo.Connection(pos1, pos1, synapse=1e-1)
            nengo.Connection(stim1, pos1, transform=50, synapse=1e-1)

            # # integrator 2
            pos2 = nengo.Ensemble(n_neurons=500, dimensions=1, radius=SCALE)
            nengo.Connection(pos2, pos2, synapse=1e-1)
            nengo.Connection(stim2, pos2, transform=50, synapse=1e-1)

            # angle of the slop, [x diff (constant), y diff (height)]
            axis_diff = nengo.Ensemble(n_neurons=1000, dimensions=2, radius=200)

            d_stim = nengo.Node([self.dist])  # x distance is constant
            nengo.Connection(d_stim, axis_diff[0])

            nengo.Connection(pos1, axis_diff[1], synapse=0.1)
            nengo.Connection(pos2, axis_diff[1], transform=-1, synapse=0.1)

            # find angle
            y_rot = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=np.pi)
            nengo.Connection(axis_diff, y_rot, function=angle)

            self.probe = nengo.Probe(y_rot, synapse=0.1)

        self.sim = nengo.Simulator(self.model, dt=sim_dt)

    def update(self, val1, val2):
        self.values1.append(val1)
        self.values2.append(val2)
        self.sim.step()

    def get_xy(self):
        return self.sim.trange(), self.sim.data[self.probe]
