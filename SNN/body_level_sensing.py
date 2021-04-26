import matplotlib.pyplot as plt
import os
import nengo

from SNN.networks.orientation_network import OrientationNetwork


class BodyOrientationModel:
    def __init__(self, wide0_points, long0_points, debug, frame_skip):
        self.frame_skip = frame_skip
        self.history = []
        self.model = nengo.Network()
        self.wideh_change = [0, 0]
        self.longh_change = [0, 0]

        pwide1, pwide2 = wide0_points
        plong1, plong2 = long0_points
        widew_diff = (pwide1 - pwide2)[0],
        longw_diff = (plong1 - plong2)[0]
        with self.model:
            self.wide_stim = nengo.Node(lambda t: self.wideh_change)
            self.long_stim = nengo.Node(lambda t: self.longh_change)

            self.wide_angle = OrientationNetwork(n_neurons=500, starting_heights=[p[-1] for p in wide0_points],
                                              width_diff=widew_diff, label='wide_angle', debug_figs=debug)
            self.long_angle = OrientationNetwork(n_neurons=500, starting_heights=[p[-1] for p in long0_points],
                                              width_diff=longw_diff, label='long_angle', debug_figs=debug)

            nengo.Connection(self.wide_stim, self.wide_angle.input, synapse=None)
            nengo.Connection(self.long_stim, self.long_angle.input, synapse=None)
            self.probe_wide = nengo.Probe(self.wide_angle.output, synapse=0.1)
            self.probe_long = nengo.Probe(self.long_angle.output, synapse=0.1)
        self.sim = nengo.Simulator(self.model, 0.001)

    def update(self, valx, valy):
        self.xh_change = valx
        self.yh_change = valy
        self.sim.step()

    def get_xy(self, probe=None):
        if probe is None:
            probe = self.probe_wide
        return self.sim.trange(), self.sim.data[probe]

    @property
    def curr_val(self):
        return self.sim.data[self.probe_wide][-1][0], self.sim.data[self.probe_long][-1][0]

    def save_figs(self, axis):

        assert axis in ['wide', 'long']
        path = os.path.dirname(__file__)

        if axis == 'wide':
            angle = self.wide_angle
        if axis == 'long':
            angle = self.long_angle

        for name, probe in angle.probes_dict.items():
            figname = angle.label + '_' + name + '.png'
            plt.figure()
            plt.title(figname)
            plt.grid()
            plt.plot(*self.get_xy(probe))
            plt.savefig(os.path.join(path, 'out', figname))
