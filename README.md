# Raksa 🛡️

[![PyPI Version](https://img.shields.io/pypi/v/raksa?color=blue&label=PyPI)](https://pypi.org/project/raksa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Raksa-black?logo=github)](https://github.com/Muhammad-Ikhwan-Fathulloh/Raksa)
[![Python](https://img.shields.io/pypi/pyversions/raksa)](https://pypi.org/project/raksa/)

**Raksa** (derived from Sanskrit/Indonesian word for *Protector* or *Guardian*) is a high-persistence, self-contained MicroPython library built for the **Noc Lab** TinyML ecosystem as part of the **'Tiny Chip, Big Brain'** workshop. It serves as an end-to-end *In-situ Analytics* gateway, securing stable asynchronous telemetry data stream from edge devices (ESP32/RP2040) to a FastAPI cloud backend.

Leveraging native compiler execution (`@micropython.native`) for fast on-device inference and memory consolidations (`gc.collect()`), Raksa protects MicroPython devices from heap fragmentation during high-velocity TinyML operations.

---

## Key Features
- 🔄 **Persistent WebSockets**: Pure python RFC 6455 client implementation utilizing raw asynchronous `uasyncio` streams without external footprint overhead.
- ⚡ **High-Speed In-situ Analytics**: Edge inference calculations optimized directly via `@micropython.native` decorators to ensure minimal loop latency.
- 🧹 **Heap Consolidation**: Garbage collection executed dynamically inside data sync cycles to prevent RAM fragmentation on constrained hardware.
- 🇮🇩 **Nusantara Resiliency**: Promotes self-reliance, stability, and secure data guardianship for resource-constrained edge-to-cloud architectures.

---

## Installation

### Via `pip` (CPython / Desktop Testing)
```bash
pip install raksa
```

### Via `mip` (MicroPython on Device)

#### 1. Using `mpremote` (Recommended)
Connect your microcontroller board to your pc via USB/Serial and run:
```bash
mpremote mip install github:Muhammad-Ikhwan-Fathulloh/Raksa
```

#### 2. Directly in MicroPython REPL
Ensure your microcontroller board has an active Wi-Fi connection, then run:
```python
import mip
mip.install("github:Muhammad-Ikhwan-Fathulloh/Raksa")
```

---

## Minimalist Code Example (ESP32 / RP2040)

Below is an end-to-end usage code highlighting edge inference and data synchronization under 10 lines of functional code:

```python
import uasyncio as asyncio
from raksa import RaksaClient

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    model = ([[0.5, -0.2], [0.1, 0.9]], [0.1, 0.0]) # TinyML weights & biases
    inputs = [0.8, 1.5] # Raw sensor variables
    
    pred = client.infer(model, inputs)
    await client.sync({"model": "NocML_V1", "features": inputs, "prediction": pred})

asyncio.run(main())
```

---

## Complete Examples & Machine Learning Features

Full production-ready examples are available under the [`examples/`](examples/) folder, covering both hardware-specific scenarios and internal, Scikit-Learn-compatible Machine Learning algorithms:

### Hardware-Connected Scenarios
| File                                                    | Scenario                                   | Sensors                    |
| ------------------------------------------------------- | ------------------------------------------ | -------------------------- |
| [`main_edge.py`](examples/main_edge.py)                 | Quick-start minimalist (< 10 lines)        | —                          |
| [`esp32_wifi_sensor.py`](examples/esp32_wifi_sensor.py) | ADC analog sensor + 3-class classification | LDR / MQ-x / Potentiometer |
| [`esp32_dht_monitor.py`](examples/esp32_dht_monitor.py) | Temperature & humidity anomaly detection   | DHT11 / DHT22              |

### On-Device Machine Learning & TinyML
| File                                                    | Algorithms Covered               | API Classes                                                         |
| ------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------- |
| [`ml_preprocessing.py`](examples/ml_preprocessing.py)   | Custom Data Scaling & Extensions | `MinMaxScaler`, `StandardScaler`, `PolynomialFeatures`              |
| [`ml_classification.py`](examples/ml_classification.py) | Optimized Classifiers            | `KNN`, `NaiveBayes`, `LogisticRegression`, `DecisionTreeClassifier` |
| [`ml_clustering.py`](examples/ml_clustering.py)         | Unsupervised Grouping / Fit      | `KMeans`                                                            |
| [`ml_forecasting.py`](examples/ml_forecasting.py)       | Simple Time Series forecasting   | `LinearForecaster`                                                  |
| [`ml_neural.py`](examples/ml_neural.py)                 | Perceptron & Feedforward NN      | `Perceptron`, `TinyNeuralNetwork`, `Activation`                     |
| [`ml_evaluation.py`](examples/ml_evaluation.py)         | Evaluation & Split Metrics       | `LabelEncoder`, `TrainTestSplit`, `ConfusionMatrix`                 |
| [`ml_anomaly.py`](examples/ml_anomaly.py)               | Anomaly Detection & PCA          | `AnomalyDetector`, `PCA`                                            |

---

## Machine Learning Reference

Raksa bundles a highly optimized port of the **NocML** C++ library along with original advanced TinyML tools, utilizing `@micropython.native` compiled mathematical loops to secure lightning-fast predictions directly on MicroPython edge boards without external dependencies.

### Preprocessing & Decomposition
- **`MinMaxScaler(dims, min_vals, max_vals)`**: Rescales variables to a range `[0.0 - 1.0]`.
- **`StandardScaler(dims, means, stddevs)`**: Standardizes features using population mean and variance.
- **`PolynomialFeatures(degree)`**: Expands input dimension using combinations with replacement.
- **`PCA(n_components)`**: Principal Component Analysis for dimensionality reduction with `fit(X, n_samples, dims)` and `transform(x)`.
- **`LabelEncoder()`**: Standard encoder for string labels to integer classes with `fit()`, `encode()`, and `decode()`.

### Neural Networks
- **`Activation`**: Collection of activation functions: `sigmoid(x)`, `relu(x)`, `tanh(x)`, and `softmax(x)`.
- **`Perceptron(dims, lr=0.01)`**: Single-layer perceptron for binary classification with `train(X, y)` and `predict(x)`.
- **`TinyNeuralNetwork(layer_sizes, lr=0.1)`**: Feedforward neural network supporting arbitrary hidden layers and backpropagation `train(X, y, epochs)`.

### Classification, Clustering & Forecasting
- **`KNN(training_data, labels, num_samples, dims, k=3)`**: Traditional classification using Euclidean distances.
- **`NaiveBayes(num_classes, dims, means, vars, priors)`**: Gaussian probabilistic classification.
- **`LogisticRegression(dims, weights, bias)`**: Fast binary classification with `predict()` and `predict_proba()`.
- **`DecisionTreeClassifier(nodes, num_nodes)`**: Node list traverse trees supporting dictionary, tuple, or custom node structures.
- **`KMeans(k, dims, centroids)`**: Clusters features into centroids. Supports `run(data, num_samples)` for on-device fitting.
- **`LinearForecaster()`**: Computes simple linear regressive trends ($y=mx+c$) natively via `fit(x, y)` and `forecastNext()`.

### Validation & Evaluation
- **`AnomalyDetector(threshold=2.5)`**: Z-score outlier detector with `fit(X, n_samples, dims)` and `detect(x)`.
- **`TrainTestSplit.split(X, y, dims, test_ratio)`**: Shuffles and splits tabular data into train/test sets using LCG.
- **`ConfusionMatrix.report(y_true, y_pred)`**: Compares predictions and outputs accuracy, precision, recall, and F1 metrics.


---

## Acknowledgments & Credits

The machine learning capabilities contained in Raksa are ported from the [NocML C++ Library for Arduino](https://github.com/Nocturnailed-Community/NocML), developed by Muhammad Ikhwan Fathulloh. Special credits go to the original creators for providing the foundation of these resource-constrained edge execution algorithms.

---

## Licenses & Copyrights
Developed by **Noc Lab** for nurturing TinyML literacy on hardware developers. Licensed under the MIT License.