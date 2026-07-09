# Raksa 🛡️

[![PyPI Version](https://img.shields.io/pypi/v/raksa?color=blue&label=PyPI)](https://pypi.org/project/raksa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Raksa-black?logo=github)](https://github.com/Muhammad-Ikhwan-Fathulloh/Raksa)
[![Python](https://img.shields.io/pypi/pyversions/raksa)](https://pypi.org/project/raksa/)

[English Version](README.md)

**Raksa** (diambil dari bahasa Sansekerta/Indonesia yang berarti *Pelindung* atau *Penjaga*) adalah library MicroPython persistensi-tinggi yang dirancang khusus untuk ekosistem **Noc Lab** dalam workshop **'Tiny Chip, Big Brain'**. Library ini bertindak sebagai jembatan *In-situ Analytics* ujung-ke-ujung (end-to-end) yang mengamankan aliran data telemetry dari perangkat edge (ESP32/RP2040) menuju FastAPI cloud backend secara asinkron.

Dengan performa komputasi native (`@micropython.native`) untuk inferensi model *on-device* dan mekanisme pembersihan memori agresif (`gc.collect()`), Raksa melestarikan stabilitas RAM perangkat mikro Anda dari fragmentasi heap di kala menjalankan beban kerja TinyML secara terus-menerus.

---

## Fitur Utama
- 🔄 **Persistensi WebSocket Mandiri**: Handshake RFC 6455 internal berbasis soket asinkron `uasyncio` tanpa membebani memori dengan library eksternal.
- ⚡ **In-situ Analytics Cepat**: Eksekusi algoritma inferensi TinyML didukung akselerasi `@micropython.native` untuk meminimalkan latensi.
- 🧹 **Konsolidasi Memori Dinamis**: Pemanggilan Garbage Collector terarah di setiap siklus pengiriman mencegah fragmentasi RAM/out-of-memory.
- 🇮🇩 **Arsitektur Nusantara Tangguh**: Konsep penjaga data yang mengedepankan efisiensi, ketangguhan mandiri, dan integrasi erat edge-to-cloud.

---

## Cara Instalasi

### Via `pip` (CPython / Pengujian Desktop)
```bash
pip install raksa
```

### Via `mip` (MicroPython di Perangkat)

#### 1. Menggunakan `mpremote` (Direkomendasikan)
Hubungkan perangkat Anda ke PC via USB, kemudian jalankan perintah berikut:
```bash
mpremote mip install github:Muhammad-Ikhwan-Fathulloh/Raksa
```

#### 2. Memasang langsung via REPL MicroPython
Pastikan board Anda memiliki koneksi Wi-Fi yang aktif, lalu jalankan perintah berikut di REPL:
```python
import mip
mip.install("github:Muhammad-Ikhwan-Fathulloh/Raksa")
```

---

## Contoh Kode Penggunaan Minimalis (ESP32)

Berikut adalah contoh skenario In-situ Analytics & sinkronisasi data ke cloud backend dalam kurang dari 10 baris kode fungsional:

```python
import uasyncio as asyncio
from raksa import RaksaClient

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    model = ([[0.5, -0.2], [0.1, 0.9]], [0.1, 0.0]) # Matriks weight & bias TinyML
    inputs = [0.8, 1.5] # Data sensor mentah
    
    pred = client.infer(model, inputs)
    await client.sync({"model": "NocML_V1", "features": inputs, "prediction": pred})

asyncio.run(main())
```

---

## Contoh Lengkap & Fitur Machine Learning

Contoh siap-pakai tersedia di folder [`examples/`](examples/), mencakup skenario integrasi sensor hardware dan penggunaan algoritma Machine Learning yang kompatibel dengan Scikit-Learn:

### Skenario Koneksi Perangkat Keras
| File                                                    | Skenario                                | Sensor                     |
| ------------------------------------------------------- | --------------------------------------- | -------------------------- |
| [`main_edge.py`](examples/main_edge.py)                 | Quick-start minimalis (< 10 baris)      | —                          |
| [`esp32_wifi_sensor.py`](examples/esp32_wifi_sensor.py) | Sensor analog ADC + Klasifikasi 3 Kelas | LDR / MQ-x / Potensiometer |
| [`esp32_dht_monitor.py`](examples/esp32_dht_monitor.py) | Deteksi anomali suhu & kelembaban       | DHT11 / DHT22              |

### Machine Learning & TinyML On-Device
| File                                                    | Algoritma Yang Tersedia           | Kelas API                                                           |
| ------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------- |
| [`ml_preprocessing.py`](examples/ml_preprocessing.py)   | Penskalaan & Ekstensi Data Sensor | `MinMaxScaler`, `StandardScaler`, `PolynomialFeatures`              |
| [`ml_classification.py`](examples/ml_classification.py) | Model Klasifikasi Teroptimasi     | `KNN`, `NaiveBayes`, `LogisticRegression`, `DecisionTreeClassifier` |
| [`ml_clustering.py`](examples/ml_clustering.py)         | Pengelompokan Data (Unsupervised) | `KMeans`                                                            |
| [`ml_forecasting.py`](examples/ml_forecasting.py)       | Peramalan Tren Deret Waktu        | `LinearForecaster`                                                  |
| [`ml_neural.py`](examples/ml_neural.py)                 | Percepton & Jaringan Saraf        | `Perceptron`, `TinyNeuralNetwork`, `Activation`                     |
| [`ml_evaluation.py`](examples/ml_evaluation.py)         | Pembagian Data & Evaluasi Metrik  | `LabelEncoder`, `TrainTestSplit`, `ConfusionMatrix`                 |
| [`ml_anomaly.py`](examples/ml_anomaly.py)               | Deteksi Anomali & PCA             | `AnomalyDetector`, `PCA`                                            |

---

## Panduan Referensi Machine Learning

Raksa mengintegrasikan porting dari library C++ **NocML** serta algoritma tingkat lanjut TinyML orisinal yang telah dioptimalkan secara mendalam menggunakan akselerasi `@micropython.native` kelas-kompilasi untuk memproses prediksi cepat secara lokal pada papan edge MicroPython tanpa beban memori tinggi ataupun dependensi berat.

### Preprocessing & Dekomposisi
- **`MinMaxScaler(dims, min_vals, max_vals)`**: Menyesuaikan skala fitur ke rentang `[0.0 - 1.0]`.
- **`StandardScaler(dims, means, stddevs)`**: Menstandardisasikan fitur menggunakan populasi mean dan varians (mean=0, std=1).
- **`PolynomialFeatures(degree)`**: Mengekspansi dimensi fitur menjadi kombinasi orde terpilih.
- **`PCA(n_components)`**: Principal Component Analysis untuk reduksi dimensi dengan `fit(X, n_samples, dims)` dan `transform(x)`.
- **`LabelEncoder()`**: Pengodean label string menjadi indeks kelas integer dengan fungsi `fit()`, `encode()`, dan `decode()`.

### Jaringan Saraf (Neural Networks)
- **`Activation`**: Kumpulan fungsi aktivasi: `sigmoid(x)`, `relu(x)`, `tanh(x)`, dan `softmax(x)`.
- **`Perceptron(dims, lr=0.01)`**: Single-layer perceptron untuk klasifikasi biner dengan `train(X, y)` dan `predict(x)`.
- **`TinyNeuralNetwork(layer_sizes, lr=0.1)`**: Feedforward neural network dengan layer tersembunyi fleksibel dan latih backpropagation `train(X, y, epochs)`.

### Klasifikasi, Clustering & Peramalan
- **`KNN(training_data, labels, num_samples, dims, k=3)`**: Klasifikasi berbasis perbandingan jarak Euclidean tetangga terdekat.
- **`NaiveBayes(num_classes, dims, means, vars, priors)`**: Klasifikasi berbasis peluang Gaussian.
- **`LogisticRegression(dims, weights, bias)`**: Model keputusan biner super cepat dengan metode `predict()` dan `predict_proba()`.
- **`DecisionTreeClassifier(nodes, num_nodes)`**: Penelusuran pohon keputusan mendukung struktur list bertipe dict, tuple, atau class node.
- **`KMeans(k, dims, centroids)`**: Mengelompokkan titik data ke centroid terdekat. Mendukung fungsi `run(data, num_samples)` untuk latihan lokal langsung.
- **`LinearForecaster()`**: Melakukan fitting regresi linear sederhana ($y=mx+c$) melalui `fit(x, y)` dan memproyeksikan deret menit depan via `forecastNext()`.

### Validasi & Evaluasi
- **`AnomalyDetector(threshold=2.5)`**: Pendeteksi anomali berbasis skor outlier z-score dengan fungsi `fit(X, n_samples, dims)` dan `detect(x)`.
- **`TrainTestSplit.split(X, y, dims, test_ratio)`**: Mengacak dan membagi dataset menjadi data latih dan data uji secara mandiri.
- **`ConfusionMatrix.report(y_true, y_pred)`**: Membandingkan hasil prediksi dengan label asli dan mengeluarkan metrik akurasi, presisi, recall, dan skor F1.


---

## Apresiasi & Sumber Pengembangan

Kemampuan Machine Learning di dalam library Raksa ini diporting langsung dari [NocML C++ Library for Arduino](https://github.com/Nocturnailed-Community/NocML) yang diciptakan oleh Muhammad Ikhwan Fathulloh. Terima kasih sebesar-besarnya kepada tim pengembang asli atas kontribusi algoritma edge cerdas berkinerja tinggi ini.

---

## Lisensi & Hak Cipta
Dikembangkan oleh **Noc Lab** untuk mendukung edukasi perangkat keras TinyML di Indonesia. Dilindungi oleh Lisensi MIT.