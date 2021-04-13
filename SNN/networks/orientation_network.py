import numpy as np
import nengo
from nengo.exceptions import ObsoleteError

from SNN.networks.integrator_array import IntegratorArray


class OrientationNetwork(nengo.Network):
    def __init__(self, n_neurons, width_diff, starting_heights=None, debug_figs=False, **kwargs):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Orientation")
        super().__init__(**kwargs)
        self.w_diff = width_diff
        self.probes_dict = {}
        with self:
            # change in width and height
            self.integrators = IntegratorArray(n_neurons=n_neurons,
                                               n_ensembles=2,
                                               recurrent_tau=0.08,
                                               inp_transform=50,
                                               inp_synapse=None,
                                               ens_dimensions=1,
                                               radius=0.1)
            self.input = self.integrators.input

            # difference of h1-h2
            self.integrator_out = nengo.Ensemble(n_neurons=n_neurons, dimensions=2)
            nengo.Connection(self.integrators.output, self.integrator_out, synapse=None)

            self.h_diff = nengo.Ensemble(n_neurons=n_neurons, dimensions=1)
            nengo.Connection(self.integrator_out, self.h_diff, function=lambda p: p[0] - p[1], synapse=None)

            self.angle = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, radius=np.pi)
            nengo.Connection(self.h_diff, self.angle, function=lambda p: np.arctan2(p, self.w_diff), synapse=None)

            if debug_figs:
                self.probes_dict['h_integrator'] = nengo.Probe(self.integrators.output, synapse=0.1)
                self.probes_dict['h_diff'] = nengo.Probe(self.h_diff, synapse=0.1)
                self.probes_dict['rot_angle'] = nengo.Probe(self.angle, synapse=0.1)

        self.output = self.angle
