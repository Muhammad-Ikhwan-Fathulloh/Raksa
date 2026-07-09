# raksa/models.py
import math

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


class LinearForecaster:
    def __init__(self):
        self.m = 0.0
        self.c = 0.0
        self.last_x = 0.0
        self.step = 1.0

    def fit(self, x, y):
        n = len(x)
        if n == 0:
            return
        sum_x = sum(x)
        sum_y = sum(y)
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        num = 0.0
        den = 0.0
        for i in range(n):
            dx = x[i] - mean_x
            num += dx * (y[i] - mean_y)
            den += dx * dx
            
        if den != 0.0:
            self.m = num / den
        else:
            self.m = 0.0
        self.c = mean_y - self.m * mean_x
        
        self.last_x = x[-1]
        self.step = (x[-1] - x[0]) / (n - 1) if n > 1 else 1.0

    def predict(self, x):
        return self.m * x + self.c

    def forecastNext(self):
        return self.m * (self.last_x + self.step) + self.c


class KNN:
    def __init__(self, training_data, labels, num_samples, dims, k=3):
        self.training_data = list(training_data)
        self.labels = list(labels)
        self.num_samples = num_samples
        self.dims = dims
        self.k = k
        self.is_flat = len(self.training_data) == num_samples * dims

    @micropython.native
    def predict(self, x):
        dists = [0.0] * self.num_samples
        if self.is_flat:
            for i in range(self.num_samples):
                d2 = 0.0
                offset = i * self.dims
                for j in range(self.dims):
                    diff = x[j] - self.training_data[offset + j]
                    d2 += diff * diff
                dists[i] = d2
        else:
            for i in range(self.num_samples):
                d2 = 0.0
                row = self.training_data[i]
                for j in range(self.dims):
                    diff = x[j] - row[j]
                    d2 += diff * diff
                dists[i] = d2

        pairs = []
        for i in range(self.num_samples):
            pairs.append((dists[i], self.labels[i]))
        pairs.sort()
        
        votes = {}
        for i in range(self.k):
            lbl = pairs[i][1]
            votes[lbl] = votes.get(lbl, 0) + 1
            
        max_votes = -1
        winner = None
        for lbl, count in votes.items():
            if count > max_votes:
                max_votes = count
                winner = lbl
        return winner


class NaiveBayes:
    def __init__(self, num_classes, dims, means, vars, priors):
        self.num_classes = num_classes
        self.dims = dims
        self.means = list(means)
        self.vars = list(vars)
        self.priors = list(priors)

    @micropython.native
    def predict(self, x):
        best_class = 0
        max_log_prob = -1e9
        
        for c in range(self.num_classes):
            log_prob = math.log(self.priors[c])
            for j in range(self.dims):
                idx = c * self.dims + j
                mean_val = self.means[idx]
                var_val = self.vars[idx]
                if var_val <= 0.0:
                    var_val = 1e-9
                diff = x[j] - mean_val
                log_prob += -0.5 * math.log(6.283185307179586 * var_val) - (diff * diff) / (2.0 * var_val)
                
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = c
        return best_class


class LogisticRegression:
    def __init__(self, dims, weights, bias):
        self.dims = dims
        self.weights = list(weights)
        self.bias = bias

    @micropython.native
    def predict(self, x):
        z = self.bias
        for i in range(self.dims):
            z += self.weights[i] * x[i]
        return 1 if z >= 0.0 else 0

    def predict_proba(self, x):
        z = self.bias
        for i in range(self.dims):
            z += self.weights[i] * x[i]
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 0.0 if z < 0 else 1.0


class DecisionTreeClassifier:
    def __init__(self, nodes, num_nodes=None):
        self.nodes = list(nodes)
        self.num_nodes = len(self.nodes) if num_nodes is None else num_nodes

    @micropython.native
    def predict(self, x):
        curr_idx = 0
        while curr_idx < self.num_nodes:
            node = self.nodes[curr_idx]
            if isinstance(node, dict):
                feat = node.get("feature", -1)
                if feat == -1: feat = node.get("feature_idx", -1)
                threshold = node.get("threshold", 0.0)
                left = node.get("left", -1)
                if left == -1: left = node.get("left_child", -1)
                right = node.get("right", -1)
                if right == -1: right = node.get("right_child", -1)
                val = node.get("value", -1)
            elif isinstance(node, (list, tuple)):
                feat = int(node[0])
                threshold = float(node[1])
                left = int(node[2])
                right = int(node[3])
                val = int(node[4])
            else:
                feat = getattr(node, "feature", -1)
                if feat == -1: feat = getattr(node, "feature_idx", -1)
                threshold = getattr(node, "threshold", 0.0)
                left = getattr(node, "left", -1)
                if left == -1: left = getattr(node, "left_child", -1)
                right = getattr(node, "right", -1)
                if right == -1: right = getattr(node, "right_child", -1)
                val = getattr(node, "value", -1)

            if feat < 0:
                return val
            if x[feat] <= threshold:
                curr_idx = left
            else:
                curr_idx = right
        return -1


class KMeans:
    def __init__(self, k, dims, centroids=None):
        self.k = k
        self.dims = dims
        if centroids is not None:
            self.centroids = list(centroids)
            self.is_flat = len(self.centroids) == k * dims
        else:
            self.centroids = [0.0] * (k * dims)
            self.is_flat = True

    @micropython.native
    def predict(self, x):
        best_k = 0
        min_d2 = 1e9
        for i in range(self.k):
            d2 = 0.0
            if self.is_flat:
                offset = i * self.dims
                for j in range(self.dims):
                    diff = x[j] - self.centroids[offset + j]
                    d2 += diff * diff
            else:
                row = self.centroids[i]
                for j in range(self.dims):
                    diff = x[j] - row[j]
                    d2 += diff * diff
            if d2 < min_d2:
                min_d2 = d2
                best_k = i
        return best_k

    def run(self, data, num_samples, assignments=None, max_iters=10):
        data = list(data)
        is_data_flat = len(data) == num_samples * self.dims
        if assignments is None:
            assignments = [0] * num_samples
            
        total_zeros = True
        for val in self.centroids:
            if val != 0.0:
                total_zeros = False
                break
        if total_zeros:
            for i in range(self.k):
                sample_idx = i % num_samples
                if is_data_flat:
                    for j in range(self.dims):
                        self.centroids[i * self.dims + j] = data[sample_idx * self.dims + j]
                else:
                    self.centroids[i] = list(data[sample_idx])

        for _ in range(max_iters):
            changed = False
            for i in range(num_samples):
                if is_data_flat:
                    x = data[i * self.dims : (i + 1) * self.dims]
                else:
                    x = data[i]
                old_k = assignments[i]
                new_k = self.predict(x)
                if old_k != new_k:
                    assignments[i] = new_k
                    changed = True
            if not changed:
                break

            cnt = [0] * self.k
            sums = [0.0] * (self.k * self.dims)
            for i in range(num_samples):
                cid = assignments[i]
                cnt[cid] += 1
                if is_data_flat:
                    for j in range(self.dims):
                        sums[cid * self.dims + j] += data[i * self.dims + j]
                else:
                    for j in range(self.dims):
                        sums[cid * self.dims + j] += data[i][j]

            for cid in range(self.k):
                n_pts = cnt[cid]
                if n_pts > 0:
                    for j in range(self.dims):
                        val = sums[cid * self.dims + j] / n_pts
                        if self.is_flat:
                            self.centroids[cid * self.dims + j] = val
                        else:
                            self.centroids[cid][j] = val
        return assignments
