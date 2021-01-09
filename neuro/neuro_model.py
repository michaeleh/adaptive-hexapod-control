import nengo


class NeuroIntegrator:
    def __init__(self, sim_dt):
        self.model = nengo.Network()
        self.values = [0]

        with self.model:
            stim = nengo.Node(lambda t: self.values[-1])

            pos = nengo.Ensemble(n_neurons=100, dimensions=1, radius=65)
            nengo.Connection(pos, pos, synapse=1e-2)
            nengo.Connection(stim, pos, transform=5, synapse=1e-2)
            self.pos_probe = nengo.Probe(pos, synapse=sim_dt)

        self.sim = nengo.Simulator(self.model, dt=sim_dt)

    def update(self, val):
        self.values.append(val)
        self.sim.step()

    def get_xy(self):
        return self.sim.trange(), self.sim.data[self.pos_probe]
