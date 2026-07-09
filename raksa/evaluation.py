# raksa/evaluation.py
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


class AnomalyDetector:
    """
    Z-Score based anomaly detection for sensor data on edge devices.
    Detects outliers by comparing z-scores against a configurable threshold.
    """
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        self.means = None
        self.stds = None
        self.dims = 0

    def fit(self, X, n_samples, dims):
        import math
        self.dims = dims
        self.means = [0.0] * dims
        self.stds = [0.0] * dims

        for i in range(n_samples):
            for j in range(dims):
                self.means[j] += X[i * dims + j]
        for j in range(dims):
            self.means[j] /= n_samples

        for i in range(n_samples):
            for j in range(dims):
                diff = X[i * dims + j] - self.means[j]
                self.stds[j] += diff * diff
        for j in range(dims):
            self.stds[j] = math.sqrt(self.stds[j] / n_samples) if n_samples > 0 else 1.0
            if self.stds[j] < 1e-9:
                self.stds[j] = 1e-9

    @micropython.native
    def score(self, x):
        max_z = 0.0
        for j in range(self.dims):
            z = abs(x[j] - self.means[j]) / self.stds[j]
            if z > max_z:
                max_z = z
        return max_z

    @micropython.native
    def detect(self, x):
        return self.score(x) > self.threshold


class TrainTestSplit:
    """
    Split dataset into training and test sets without external dependencies.
    Supports reproducible pseudo-random splitting via seed.
    """

    @staticmethod
    def split(X, y, dims, test_ratio=0.2, seed=42):
        n = len(y)
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_test

        indices = list(range(n))
        s = seed
        for i in range(n - 1, 0, -1):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            j = s % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X_train = []
        y_train = []
        for i in train_idx:
            for j in range(dims):
                X_train.append(X[i * dims + j])
            y_train.append(y[i])

        X_test = []
        y_test = []
        for i in test_idx:
            for j in range(dims):
                X_test.append(X[i * dims + j])
            y_test.append(y[i])

        return X_train, y_train, X_test, y_test


class ConfusionMatrix:
    """
    Compute classification metrics from predictions and true labels.
    Provides accuracy, precision, recall, and F1-score per class.
    """

    @staticmethod
    def compute(y_true, y_pred):
        classes = {}
        for lbl in y_true:
            classes[lbl] = True
        for lbl in y_pred:
            classes[lbl] = True
        class_list = sorted(classes.keys())
        n_classes = len(class_list)
        class_idx = {}
        for i, c in enumerate(class_list):
            class_idx[c] = i

        matrix = [[0] * n_classes for _ in range(n_classes)]
        n = len(y_true)
        correct = 0
        for i in range(n):
            t = class_idx[y_true[i]]
            p = class_idx[y_pred[i]]
            matrix[t][p] += 1
            if t == p:
                correct += 1

        accuracy = correct / n if n > 0 else 0.0

        per_class = {}
        for ci, c in enumerate(class_list):
            tp = matrix[ci][ci]
            fp = sum(matrix[r][ci] for r in range(n_classes)) - tp
            fn = sum(matrix[ci][p] for p in range(n_classes)) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[c] = {"precision": precision, "recall": recall, "f1": f1}

        return {
            "accuracy": accuracy,
            "classes": class_list,
            "matrix": matrix,
            "per_class": per_class
        }

    @staticmethod
    def report(y_true, y_pred):
        result = ConfusionMatrix.compute(y_true, y_pred)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 48)
        for c in result["classes"]:
            m = result["per_class"][c]
            print(f"{str(c):<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
