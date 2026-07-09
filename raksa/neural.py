# raksa/neural.py
import gc

try:
    import micropython
except ImportError:
    class micropython:
        @staticmethod
        def native(f):
            return f
        @staticmethod
        def viper(f):
            return f


class Activation:
    """Collection of activation functions optimized for TinyML inference."""

    @staticmethod
    @micropython.native
    def sigmoid(x):
        import math
        if isinstance(x, (list, tuple)):
            out = [0.0] * len(x)
            for i in range(len(x)):
                v = x[i]
                if v > 20.0:
                    out[i] = 1.0
                elif v < -20.0:
                    out[i] = 0.0
                else:
                    out[i] = 1.0 / (1.0 + math.exp(-v))
            return out
        if x > 20.0:
            return 1.0
        if x < -20.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    @micropython.native
    def relu(x):
        if isinstance(x, (list, tuple)):
            out = [0.0] * len(x)
            for i in range(len(x)):
                out[i] = x[i] if x[i] > 0.0 else 0.0
            return out
        return x if x > 0.0 else 0.0

    @staticmethod
    @micropython.native
    def tanh(x):
        import math
        if isinstance(x, (list, tuple)):
            out = [0.0] * len(x)
            for i in range(len(x)):
                out[i] = math.tanh(x[i])
            return out
        return math.tanh(x)

    @staticmethod
    @micropython.native
    def softmax(x):
        import math
        n = len(x)
        max_v = x[0]
        for i in range(1, n):
            if x[i] > max_v:
                max_v = x[i]
        exps = [0.0] * n
        total = 0.0
        for i in range(n):
            e = math.exp(x[i] - max_v)
            exps[i] = e
            total += e
        for i in range(n):
            exps[i] /= total
        return exps


class Perceptron:
    """
    Single-layer Perceptron for binary classification on edge devices.
    Supports on-device training with adjustable learning rate and epochs.
    """
    def __init__(self, dims, lr=0.01):
        self.dims = dims
        self.lr = lr
        self.weights = [0.0] * dims
        self.bias = 0.0

    @micropython.native
    def predict_raw(self, x):
        z = self.bias
        for i in range(self.dims):
            z += self.weights[i] * x[i]
        return z

    @micropython.native
    def predict(self, x):
        return 1 if self.predict_raw(x) >= 0.0 else 0

    def train(self, X, y, epochs=100):
        """
        Train on flat data: X is flat list [x0_f0, x0_f1, ..., x1_f0, ...], y is labels list.
        """
        n = len(y)
        for _ in range(epochs):
            for i in range(n):
                offset = i * self.dims
                sample = X[offset:offset + self.dims]
                pred = self.predict(sample)
                error = y[i] - pred
                if error != 0:
                    for j in range(self.dims):
                        self.weights[j] += self.lr * error * sample[j]
                    self.bias += self.lr * error


class TinyNeuralNetwork:
    """
    Multi-layer feedforward neural network with backpropagation training.
    Designed for lightweight on-device learning on MicroPython boards.
    Supports configurable hidden layers, sigmoid activation, and MSE loss.
    """
    def __init__(self, layer_sizes, lr=0.1):
        """
        Args:
            layer_sizes: list of ints, e.g. [2, 4, 1] for 2-input, 4-hidden, 1-output.
            lr: learning rate.
        """
        self.layer_sizes = list(layer_sizes)
        self.lr = lr
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        for l in range(self.num_layers - 1):
            rows = layer_sizes[l + 1]
            cols = layer_sizes[l]
            w = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    seed = ((i * 7 + j * 13 + l * 31) % 100) / 100.0 - 0.5
                    row.append(seed)
                w.append(row)
            self.weights.append(w)
            self.biases.append([0.0] * rows)

    def _sigmoid(self, x):
        import math
        if x > 20.0:
            return 1.0
        if x < -20.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    @micropython.native
    def forward(self, x):
        """Forward pass. Returns (activations_per_layer, z_per_layer)."""
        activations = [list(x)]
        zs = []
        current = list(x)
        for l in range(self.num_layers - 1):
            w = self.weights[l]
            b = self.biases[l]
            n_out = len(w)
            z = [0.0] * n_out
            a = [0.0] * n_out
            for i in range(n_out):
                val = b[i]
                for j in range(len(current)):
                    val += w[i][j] * current[j]
                z[i] = val
                a[i] = self._sigmoid(val)
            zs.append(z)
            activations.append(a)
            current = a
        return activations, zs

    @micropython.native
    def predict(self, x):
        """Run forward pass and return output layer activations."""
        activations, _ = self.forward(x)
        return activations[-1]

    def train(self, X, y, epochs=1000):
        """
        Train with backpropagation.
        X: flat list [x0_f0, x0_f1, ..., x1_f0, ...], y: list of target lists.
        """
        n_samples = len(y)
        input_size = self.layer_sizes[0]
        for _ in range(epochs):
            for s in range(n_samples):
                sample = X[s * input_size:(s + 1) * input_size]
                target = y[s] if isinstance(y[s], (list, tuple)) else [y[s]]

                activations, zs = self.forward(sample)

                # Compute output error delta
                output = activations[-1]
                n_out = len(output)
                delta = [0.0] * n_out
                for i in range(n_out):
                    o = output[i]
                    delta[i] = (o - target[i]) * o * (1.0 - o)

                deltas = [None] * (self.num_layers - 1)
                deltas[-1] = delta

                # Backpropagate deltas
                for l in range(self.num_layers - 3, -1, -1):
                    w_next = self.weights[l + 1]
                    a = activations[l + 1]
                    n_curr = len(a)
                    n_next = len(deltas[l + 1])
                    d = [0.0] * n_curr
                    for i in range(n_curr):
                        err = 0.0
                        for j in range(n_next):
                            err += w_next[j][i] * deltas[l + 1][j]
                        d[i] = err * a[i] * (1.0 - a[i])
                    deltas[l] = d

                # Update weights and biases
                for l in range(self.num_layers - 1):
                    a_prev = activations[l]
                    n_out_l = len(deltas[l])
                    n_in_l = len(a_prev)
                    for i in range(n_out_l):
                        for j in range(n_in_l):
                            self.weights[l][i][j] -= self.lr * deltas[l][i] * a_prev[j]
                        self.biases[l][i] -= self.lr * deltas[l][i]
            gc.collect()
