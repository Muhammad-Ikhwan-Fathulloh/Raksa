# raksa/preprocessing.py
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


class MinMaxScaler:
    def __init__(self, dims, min_vals, max_vals):
        self.dims = dims
        self.min_vals = list(min_vals)
        self.max_vals = list(max_vals)
        self.ranges = [1.0] * dims
        for i in range(dims):
            r = self.max_vals[i] - self.min_vals[i]
            self.ranges[i] = r if r != 0.0 else 1.0

    @micropython.native
    def transform(self, x, out=None):
        if out is None:
            out = [0.0] * self.dims
        for i in range(self.dims):
            out[i] = (x[i] - self.min_vals[i]) / self.ranges[i]
        return out


class StandardScaler:
    def __init__(self, dims, means, stddevs):
        self.dims = dims
        self.means = list(means)
        self.stddevs = list(stddevs)
        self.scales = [1.0] * dims
        for i in range(dims):
            s = self.stddevs[i]
            self.scales[i] = s if s != 0.0 else 1.0

    @micropython.native
    def transform(self, x, out=None):
        if out is None:
            out = [0.0] * self.dims
        for i in range(self.dims):
            out[i] = (x[i] - self.means[i]) / self.scales[i]
        return out


class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
        self._cache = {}

    def _get_combinations(self, dims):
        if dims in self._cache:
            return self._cache[dims]

        combos = []
        for d in range(self.degree + 1):
            combos.extend(self._cvw(list(range(dims)), d))
        self._cache[dims] = combos
        return combos

    def _cvw(self, items, r):
        if r == 0:
            return [()]
        if not items:
            return []
        n = len(items)
        indices = [0] * r
        results = [tuple(items[i] for i in indices)]
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return results
            indices[i:] = [indices[i] + 1] * (r - i)
            results.append(tuple(items[k] for k in indices))

    @micropython.native
    def transform(self, x, dims=None, out=None):
        if dims is None:
            dims = len(x)
        combos = self._get_combinations(dims)
        
        n_out = len(combos)
        if out is None:
            out = [0.0] * n_out
            
        for i in range(n_out):
            combo = combos[i]
            val = 1.0
            for idx in combo:
                val *= x[idx]
            out[i] = val
        return out


class LabelEncoder:
    """
    Encode string/object labels to integer indices and decode back.
    Lightweight alternative to sklearn LabelEncoder for MicroPython.
    """
    def __init__(self):
        self.label_to_idx = {}
        self.idx_to_label = []

    def fit(self, labels):
        """Fit encoder on a list of labels."""
        seen = {}
        idx = 0
        for lbl in labels:
            if lbl not in seen:
                seen[lbl] = idx
                idx += 1
        self.label_to_idx = seen
        self.idx_to_label = [None] * len(seen)
        for lbl, i in seen.items():
            self.idx_to_label[i] = lbl
        return self

    def encode(self, labels):
        """Encode list of labels to integer indices."""
        return [self.label_to_idx[lbl] for lbl in labels]

    def decode(self, indices):
        """Decode list of integer indices back to labels."""
        return [self.idx_to_label[i] for i in indices]

    def encode_one(self, label):
        return self.label_to_idx[label]

    def decode_one(self, idx):
        return self.idx_to_label[idx]


class PCA:
    """
    Principal Component Analysis for dimensionality reduction on edge.
    Computes eigenvectors via power iteration (no numpy required).
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X, n_samples, dims):
        """
        Fit PCA on flat data X.
        X: flat list [x0_f0, x0_f1, ..., x1_f0, ...].
        """
        # Compute mean
        self.mean = [0.0] * dims
        for i in range(n_samples):
            for j in range(dims):
                self.mean[j] += X[i * dims + j]
        for j in range(dims):
            self.mean[j] /= n_samples

        # Compute covariance matrix
        cov = [[0.0] * dims for _ in range(dims)]
        for i in range(n_samples):
            for j in range(dims):
                for k in range(j, dims):
                    val = (X[i * dims + j] - self.mean[j]) * (X[i * dims + k] - self.mean[k])
                    cov[j][k] += val
                    if j != k:
                        cov[k][j] += val
        for j in range(dims):
            for k in range(dims):
                cov[j][k] /= (n_samples - 1) if n_samples > 1 else 1

        # Power iteration to find top eigenvectors
        import math
        self.components = []
        for comp in range(self.n_components):
            # Initialize vector
            vec = [1.0 / (i + 1 + comp) for i in range(dims)]
            for _ in range(50):
                # Matrix-vector multiply
                new_vec = [0.0] * dims
                for j in range(dims):
                    s = 0.0
                    for k in range(dims):
                        s += cov[j][k] * vec[k]
                    new_vec[j] = s
                # Normalize
                norm = 0.0
                for j in range(dims):
                    norm += new_vec[j] * new_vec[j]
                norm = math.sqrt(norm)
                if norm < 1e-10:
                    break
                for j in range(dims):
                    new_vec[j] /= norm
                vec = new_vec

            self.components.append(list(vec))
            # Deflate covariance matrix
            eigen_val = 0.0
            for j in range(dims):
                s = 0.0
                for k in range(dims):
                    s += cov[j][k] * vec[k]
                eigen_val += s * vec[j]
            for j in range(dims):
                for k in range(dims):
                    cov[j][k] -= eigen_val * vec[j] * vec[k]

    @micropython.native
    def transform(self, x):
        """Transform a single sample x (list) to reduced dimensions."""
        out = [0.0] * self.n_components
        for c in range(self.n_components):
            s = 0.0
            comp = self.components[c]
            for j in range(len(x)):
                s += comp[j] * (x[j] - self.mean[j])
            out[c] = s
        return out

