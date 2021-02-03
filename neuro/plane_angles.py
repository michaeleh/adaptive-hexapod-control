import pickle

import numpy as np
import math
import nengo

SCALE = 250


def angle(inp):
    x, y = inp
    return math.atan(y / x)


class PlaneRotation:
    def __init__(self, sim_dt):
        self.model = nengo.Network()
        self.values1 = [[0, 0]]
        self.values2 = [[0, 0]]  # 2d

        with self.model:
            stim1 = nengo.Node(lambda t: self.values1[-1])
            stim2 = nengo.Node(lambda t: self.values2[-1])

            # integrator 1
            pos1 = nengo.Ensemble(n_neurons=1500, dimensions=2, radius=SCALE)
            nengo.Connection(pos1, pos1, synapse=1e-1)
            nengo.Connection(stim1, pos1, transform=50, synapse=1e-1)

            # integrator 2
            pos2 = nengo.Ensemble(n_neurons=1500, dimensions=2, radius=SCALE)
            nengo.Connection(pos2, pos2, synapse=1e-1)
            nengo.Connection(stim2, pos2, transform=50, synapse=1e-1)

            # angle of the slop, [x diff (constant), y diff (height)]
            axis_diff = nengo.Ensemble(n_neurons=1500, dimensions=2, radius=SCALE)

            nengo.Connection(pos1[0], axis_diff[0], synapse=0.1)
            nengo.Connection(pos2[0], axis_diff[0], transform=-1, synapse=0.1)

            nengo.Connection(pos1[1], axis_diff[1], synapse=0.1)
            nengo.Connection(pos2[1], axis_diff[1], transform=-1, synapse=0.1)

            # find angle
            y_rot = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=np.pi)
            nengo.Connection(axis_diff, y_rot,synapse=0.1, function=angle)

            self.probe = nengo.Probe(y_rot, synapse=0.1)

        self.sim = nengo.Simulator(self.model, dt=sim_dt)

    def update(self, val1, val2):
        self.values1.append(val1)
        self.values2.append(val2)
        self.sim.step()

    def get_xy(self):
        return self.sim.trange(), self.sim.data[self.probe]

    @property
    def curr_val(self):
        return self.sim.data[self.probe][-1]
