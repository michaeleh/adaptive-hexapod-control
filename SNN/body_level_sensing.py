import pickle

import matplotlib.pyplot as plt
import os
import nengo

from SNN.orientation_network import OrientationNetwork


class BodyOrientationModel:
    def __init__(self, sim_dt, frame_skip):
        self.frame_skip = frame_skip
        self.history = []
        self.model = nengo.Network()
        with self.model:
            self.x_angle = OrientationNetwork(n_neurons=300, label='x_angle', debug_figs=True)
            self.y_angle = OrientationNetwork(n_neurons=300, label='y_angle')
            self.probe_x = nengo.Probe(self.x_angle.p1_integrator.output, synapse=0.01)
            self.probe_y = nengo.Probe(self.y_angle.output, synapse=0.01)
        self.sim = nengo.Simulator(self.model, dt=sim_dt / frame_skip)

    def update(self, valx, valy):
        self.x_angle.update(*valx)

        self.y_angle.update(*valy)
        for _ in range(self.frame_skip):
            self.sim.step()

    def get_xy(self, probe=None):
        if probe is None:
            probe = self.probe_x
        return self.sim.trange(), self.sim.data[probe]

    @property
    def curr_val(self):
        return self.sim.data[self.probe_x][-1][0], self.sim.data[self.probe_y][-1][0]

    def save_figs(self, axis):

        assert axis in ['x', 'y']
        path = os.path.dirname(__file__)

        if axis == 'x':
            angle = self.x_angle
        if axis == 'y':
            angle = self.y_angle

        for name, probe in angle.probes_dict.items():
            figname = angle.label + '_' + name + '.png'
            plt.figure()
            plt.title(figname)
            plt.grid()
            plt.plot(*self.get_xy(probe))
            plt.savefig(os.path.join(path, 'out', figname))
