import numpy as np
import nengo
from nengo.exceptions import ObsoleteError


class ProbeVsValue:
    def __init__(self, probe, callback):
        self.callback = callback
        self.probe = probe
        self.real_values = []

    def update(self, *args):
        self.real_values.append(self.callback(*args))


class IntegratorNetwork(nengo.Network):
    def __init__(self, recurrent_tau, inp_transform, inp_synapse, n_neurons, dimensions, radius, **kwargs):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Integrator")
        super().__init__(**kwargs)

        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.ensemble = nengo.Ensemble(n_neurons, dimensions=dimensions, radius=radius)
            nengo.Connection(self.ensemble, self.ensemble, synapse=recurrent_tau)
            nengo.Connection(
                self.input, self.ensemble, transform=inp_transform, synapse=inp_synapse
            )

        self.output = self.ensemble


class OrientationNetwork(nengo.Network):
    def __init__(self, n_neurons, w_diff, debug_figs=False, **kwargs):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Orientation")
        super().__init__(**kwargs)
        self.h1, self.h2 = 0, 0
        self.w_diff = w_diff
        self.probes_dict = {}
        with self:
            # change in width and height
            self.h1dt = nengo.Node(lambda t: self.h1)
            self.h2dt = nengo.Node(lambda t: self.h2)

            # integrators
            self.h1_integrator = IntegratorNetwork(recurrent_tau=0.05, inp_transform=0.01, inp_synapse=0.01,
                                                   n_neurons=n_neurons, radius=0.1, dimensions=1)
            self.h2_integrator = IntegratorNetwork(recurrent_tau=0.05, inp_transform=0.01, inp_synapse=0.01,
                                                   n_neurons=n_neurons, radius=0.1, dimensions=1)

            nengo.Connection(self.h1dt, self.h1_integrator.input)
            nengo.Connection(self.h2dt, self.h2_integrator.input)

            # difference of h1-h2
            self.h_diff = nengo.Ensemble(n_neurons=n_neurons, dimensions=1)

            nengo.Connection(self.h1_integrator.output, self.h_diff, synapse=0.1)
            nengo.Connection(self.h2_integrator.output, self.h_diff, transform=-1, synapse=0.1)

            self.angle = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, radius=np.pi)

            nengo.Connection(self.h_diff, self.angle, function=lambda p: np.arctan2(p, self.w_diff), synapse=0.1)

            if debug_figs:
                self.probes_dict['h1integrator'] = nengo.Probe(self.h1_integrator.output, synapse=0.1)
                self.probes_dict['h2integrator'] = nengo.Probe(self.h2_integrator.output, synapse=0.1)
                self.probes_dict['h_diff'] = nengo.Probe(self.h_diff, synapse=0.1)
                self.probes_dict['rot_angle'] = nengo.Probe(self.angle, synapse=0.1)

        self.output = self.angle

    def update(self, val1, val2):
        self.h1, self.h2 = val1, val2
