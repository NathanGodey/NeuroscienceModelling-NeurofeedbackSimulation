import numpy as np

class MSN_Units:
    def __init__(self, nb_units, w_init):
        self.weights = w_init * np.ones(nb_units)
        self.state = np.zeros(nb_units)
        self.nb_units = nb_units
        self.history = []
        self.update_state()

    def update_state(self):
        unsqueeze_state = np.expand_dims(self.state,1)
        if len(self.history) == 0:
            self.history = unsqueeze_state
        else:
            self.history = np.hstack((self.history,unsqueeze_state))
        if self.history.shape[1] > 1024:
            self.history = np.delete(self.history, 0, 1)
        rand = np.random.random(self.nb_units)
        self.state = (rand<self.weights).astype(int)

    def update_weight(self, feedback=None):
        if feedback is not None:
            self.weights += np.sum(self.history, axis=1) * feedback
        else:
            self.weights -= np.mean(self.history, axis=1)

    def normalize(self, norm, ratio = 10):
        self.weights = np.minimum(np.ones_like(self.weights), ratio * self.weights / norm)
