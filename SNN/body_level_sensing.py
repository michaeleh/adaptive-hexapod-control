import matplotlib.pyplot as plt
import os
import nengo

from SNN.networks.orientation_network import OrientationNetwork


class BodyOrientationModel:
    def __init__(self, x0_points, y0_points, sim_dt, frame_skip):
        self.frame_skip = frame_skip
        self.history = []
        self.model = nengo.Network()
        self.xh_change = [0, 0]
        self.yh_change = [0, 0]

        px1, px2 = x0_points
        py1, py2 = y0_points
        xw_diff = (px1 - px2)[0],
        yw_diff = (py1 - py2)[0]
        with self.model:
            self.x_stim = nengo.Node(lambda t: self.xh_change)
            self.y_stim = nengo.Node(lambda t: self.yh_change)

            self.x_angle = OrientationNetwork(n_neurons=500, starting_heights=[p[-1] for p in x0_points],
                                              width_diff=xw_diff, label='x_angle', debug_figs=True)
            self.y_angle = OrientationNetwork(n_neurons=500, starting_heights=[p[-1] for p in y0_points],
                                              width_diff=yw_diff, label='y_angle', debug_figs=False)

            nengo.Connection(self.x_stim, self.x_angle.input, synapse=None)
            nengo.Connection(self.y_stim, self.y_angle.input, synapse=None)
            self.probe_x = nengo.Probe(self.x_angle.output, synapse=0.1)
            self.probe_y = nengo.Probe(self.y_angle.output, synapse=0.1)
        self.sim = nengo.Simulator(self.model, 0.001)

    def update(self, valx, valy):
        self.xh_change = valx
        self.yh_change = valy
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
