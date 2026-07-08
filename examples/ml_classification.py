# examples/ml_classification.py
# ============================================================================
# Raksa - Contoh Klasifikasi TinyML (KNN, NaiveBayes, LogReg, DecisionTree)
# ============================================================================
# Demo penggunaan berbagai model klasifikasi super cepat untuk deteksi status
# atau kategori kondisi berbasis sensor di perangkat edge.
# ============================================================================

from raksa import KNN, NaiveBayes, LogisticRegression, DecisionTreeClassifier

# Definisi Node untuk Decision Tree
class CustomNode:
    def __init__(self, feature_idx, threshold, left_child, right_child, value):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

def main():
    # ── 1. K-Nearest Neighbors (KNN) ─────────────────────────────────────────
    print("=== 1. K-Nearest Neighbors Classifier ===")
    # Data latihan: 4 sampel dengan 2 fitur [Moisture, Temp_scaled]
    # Label: 1 = Butuh Air, 0 = Sehat
    training_data = [
        0.2, 0.8,  # Dry, Hot (1)
        0.3, 0.7,  # Dry, Warm (1)
        0.8, 0.3,  # Wet, Cool (0)
        0.7, 0.4   # Wet, Warm (0)
    ]
    labels = [1, 1, 0, 0]
    knn = KNN(training_data, labels, num_samples=4, dims=2, k=3)
    
    # Uji prediksi sensor tanah kering
    test_reading = [0.25, 0.75]
    result = knn.predict(test_reading)
    print(f"KNN memprediksi sensor {test_reading} -> {result} (1 = Butuh Air, 0 = Sehat)\n")


    # ── 2. Gaussian Naive Bayes ──────────────────────────────────────────────
    print("=== 2. Gaussian Naive Bayes ===")
    # Klasifikasi Kualitas Udara (Kelas 0 = ASRI, Kelas 1 = BERBAHAYA)
    # Fitur: [Sensor MQ135, Sensor MQ7]
    means = [100.0, 50.0,   # Kelas 0: Rata-rata sensor rendah
             400.0, 200.0]  # Kelas 1: Rata-rata sensor tinggi
    vars = [400.0, 100.0,    # Kelas 0: Varians
            1600.0, 400.0]   # Kelas 1: Varians
    priors = [0.6, 0.4]      # Peluang prior
    
    nb = NaiveBayes(num_classes=2, dims=2, means=means, vars=vars, priors=priors)
    test_gas = [350.0, 180.0]
    result = nb.predict(test_gas)
    print(f"NaiveBayes memprediksi sensor {test_gas} -> Kelas {result} (0 = Asri, 1 = Berbahaya)\n")


    # ── 3. Logistic Regression ───────────────────────────────────────────────
    print("=== 3. Logistic Regression (Binary Decision) ===")
    # Parameter diexport dari scikit-learn untuk deteksi keputusan ON/OFF AC otomatis
    # Fitur: [Temperature, Humidity]
    weights = [0.85, 0.42]
    bias = -25.5
    log_reg = LogisticRegression(dims=2, weights=weights, bias=bias)
    
    test_room = [28.5, 75.0]  # Panas dan lembab
    prediction = log_reg.predict(test_room)
    probability = log_reg.predict_proba(test_room)
    print(f"LogReg memprediksi ruangan {test_room}:")
    print(f"  - Keputusan AC: {'ON' if prediction == 1 else 'OFF'} ({prediction})")
    print(f"  - Probabilitas ON: {probability * 100:.2f}%\n")


    # ── 4. Decision Tree Classifier ──────────────────────────────────────────
    print("=== 4. Decision Tree Classifier ===")
    # Pohon keputusan penyortiran robot sederhana untuk mendeteksi Benda Besar (1) vs Kecil (0)
    # Struktur Node: [feature_idx, threshold, left_child, right_child, leaf_value]
    # Jika feature_idx < 0 (-1), maka node tersebut adalah Leaf Node dan mengembalikan leaf_val.
    
    # Representasi 1: Menggunakan list dictionary
    tree_dict = [
        {"feature": 0, "threshold": 10.0, "left": 1, "right": 2, "value": -1},  # Root
        {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": 0}, # Leaf Kiri (Kecil)
        {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": 1}  # Leaf Kanan (Besar)
    ]
    dt_dict = DecisionTreeClassifier(tree_dict)
    print(f"Dict-Tree [ Tinggi = 7.5 ]  -> Klasifikasi: {dt_dict.predict([7.5])}")
    print(f"Dict-Tree [ Tinggi = 12.5 ] -> Klasifikasi: {dt_dict.predict([12.5])}")

    # Representasi 2: Menggunakan list CustomNode (C++ NocML style layout)
    tree_objects = [
        CustomNode(feature_idx=0, threshold=10.0, left_child=1, right_child=2, value=-1),
        CustomNode(feature_idx=-1, threshold=0.0, left_child=-1, right_child=-1, value=0),
        CustomNode(feature_idx=-1, threshold=0.0, left_child=-1, right_child=-1, value=1)
    ]
    dt_obj = DecisionTreeClassifier(tree_objects)
    print(f"Obj-Tree  [ Tinggi = 8.5 ]  -> Klasifikasi: {dt_obj.predict([8.5])}")

if __name__ == "__main__":
    main()
