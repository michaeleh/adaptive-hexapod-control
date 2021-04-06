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
    def __init__(self, recurrent_tau, inp_transform, inp_synapse, n_neurons, dimensions, **kwargs):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Integrator")
        super().__init__(**kwargs)

        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.ensemble = nengo.Ensemble(n_neurons, dimensions=dimensions)
            nengo.Connection(self.ensemble, self.ensemble, synapse=recurrent_tau)
            nengo.Connection(
                self.input, self.ensemble, transform=inp_transform, synapse=inp_synapse
            )

        self.output = self.ensemble


class OrientationNetwork(nengo.Network):
    def __init__(self, n_neurons, debug_figs=False, **kwargs):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Orientation")
        super().__init__(**kwargs)
        self.inp1 = [0, 0]
        self.inp2 = [0, 0]
        self.probes_dict = {}
        with self:
            # change in width and height
            self.p1dt = nengo.Node(lambda t: self.inp1)
            self.p2dt = nengo.Node(lambda t: self.inp2)
            # integrators
            self.p1_integrator = IntegratorNetwork(recurrent_tau=0.1, inp_transform=0.15, inp_synapse=0.1,
                                                   n_neurons=n_neurons * 2, dimensions=2)
            self.p2_integrator = IntegratorNetwork(recurrent_tau=0.1, inp_transform=0.15, inp_synapse=0.1,
                                                   n_neurons=n_neurons * 2, dimensions=2)

            nengo.Connection(self.p1dt, self.p1_integrator.input)
            nengo.Connection(self.p2dt, self.p2_integrator.input)

            # difference of h and w p1-p2
            self.p_diff = nengo.Ensemble(n_neurons=2 * n_neurons, dimensions=2, radius=2)

            nengo.Connection(self.p1_integrator.output, self.p_diff, synapse=0.1)
            nengo.Connection(self.p2_integrator.output, self.p_diff, transform=-1, synapse=0.1)

            self.angle = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, radius=np.pi)

            nengo.Connection(self.p_diff, self.angle, function=lambda x: np.arctan2(x[1], x[0]), synapse=0.1)
            if debug_figs:
                self.probes_dict['p1integrator'] = nengo.Probe(self.p1_integrator.output, synapse=0.1)
                self.probes_dict['p2integrator'] = nengo.Probe(self.p2_integrator.output, synapse=0.1)
                self.probes_dict['p_diff'] = nengo.Probe(self.p_diff, synapse=0.1)
                self.probes_dict['angle'] = nengo.Probe(self.angle, synapse=0.1)

        self.output = self.angle

    def update(self, val1, val2):
        self.inp1 = list(val1)
        self.inp2 = list(val2)
        print('inps', val1, val2)
