import numpy as np
import matplotlib.pyplot as plt
import os
import nengo

from SNN.networks.integrator_array import IntegratorArray

N_LEGS = 6


class LegsHeightsModel:
    def __init__(self, n_neurons, tau, transform, height0=np.zeros(N_LEGS)):
        self.model = nengo.Network()
        self.h_change = np.zeros(N_LEGS)
        with self.model:
            self.stim = nengo.Node(lambda t: self.h_change)
            self.starting_h_stim = nengo.Node(height0)

            self.integrators = IntegratorArray(n_neurons=n_neurons,
                                               n_ensembles=N_LEGS,
                                               recurrent_tau=tau,
                                               inp_transform=transform,
                                               inp_synapse=0.1,
                                               ens_dimensions=1,
                                               radius=0.1)

            nengo.Connection(self.stim, self.integrators.input, synapse=None)
            self.integrator_out = nengo.Ensemble(n_neurons=1000 * N_LEGS, dimensions=N_LEGS)

            nengo.Connection(self.integrators.output, self.integrator_out, synapse=None)
            nengo.Connection(self.starting_h_stim, self.integrator_out, synapse=None)
            self.probe_out = nengo.Probe(self.integrator_out, synapse=0.5)

            self.sim = nengo.Simulator(self.model)

    def update(self, h_change):
        self.h_change = h_change.copy()
        self.sim.step()

    def get_xy(self):
        return self.sim.trange(), self.sim.data[self.probe_out]

    @property
    def curr_val(self):
        return self.sim.data[self.probe_out][-1]

    def save_figs(self):
        path = os.path.dirname(__file__)
        figname = 'integrator_out.png'
        plt.figure()
        plt.title(figname)
        plt.grid()
        plt.plot(*self.get_xy())
        plt.savefig(os.path.join(path, 'out', figname))
