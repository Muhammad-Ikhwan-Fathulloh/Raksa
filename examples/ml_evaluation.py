# examples/ml_evaluation.py
# ============================================================================
# Raksa - Contoh Evaluasi Model (LabelEncoder, TrainTestSplit, ConfusionMatrix)
# ============================================================================
# Demo utilitas evaluasi ML: encoding label, pembagian dataset, dan
# laporan metrik klasifikasi lengkap (accuracy, precision, recall, F1).
# ============================================================================

from raksa import LabelEncoder, TrainTestSplit, ConfusionMatrix, KNN

def main():
    # ── 1. LabelEncoder ─────────────────────────────────────────────────────
    print("=== 1. LabelEncoder ===")
    encoder = LabelEncoder()

    labels = ["kucing", "anjing", "burung", "kucing", "burung", "anjing"]
    encoder.fit(labels)

    encoded = encoder.encode(labels)
    print(f"  Label asli : {labels}")
    print(f"  Encoded    : {encoded}")

    decoded = encoder.decode(encoded)
    print(f"  Decoded    : {decoded}")
    print(f"  Encode satu: 'burung' -> {encoder.encode_one('burung')}")
    print(f"  Decode satu: 0 -> '{encoder.decode_one(0)}'")
    print()

    # ── 2. TrainTestSplit ────────────────────────────────────────────────────
    print("=== 2. TrainTestSplit ===")
    # Dataset sederhana: 10 sampel, 2 fitur, 2 kelas
    X_flat = [
        1.0, 2.0,   # sampel 0
        1.5, 1.8,   # sampel 1
        5.0, 8.0,   # sampel 2
        8.0, 8.0,   # sampel 3
        1.0, 0.6,   # sampel 4
        9.0, 11.0,  # sampel 5
        8.0, 2.0,   # sampel 6
        10.0, 2.0,  # sampel 7
        9.0, 3.0,   # sampel 8
        6.0, 1.0,   # sampel 9
    ]
    y_labels = [0, 0, 1, 1, 0, 1, 1, 1, 1, 1]

    X_train, y_train, X_test, y_test = TrainTestSplit.split(
        X_flat, y_labels, dims=2, test_ratio=0.3, seed=42
    )

    n_train = len(y_train)
    n_test = len(y_test)
    print(f"  Total sampel: {len(y_labels)}")
    print(f"  Train: {n_train} sampel, Test: {n_test} sampel")
    print(f"  y_train: {y_train}")
    print(f"  y_test : {y_test}")
    print()

    # ── 3. ConfusionMatrix ───────────────────────────────────────────────────
    print("=== 3. ConfusionMatrix ===")

    # Gunakan KNN untuk klasifikasi pada data split
    knn = KNN(X_train, y_train, num_samples=n_train, dims=2, k=3)

    # Prediksi test set
    y_pred = []
    for i in range(n_test):
        sample = X_test[i * 2:(i + 1) * 2]
        y_pred.append(knn.predict(sample))

    print(f"  y_true: {y_test}")
    print(f"  y_pred: {y_pred}")
    print()

    # Cetak laporan evaluasi
    ConfusionMatrix.report(y_test, y_pred)
    print()

    # Akses metric secara programatik
    result = ConfusionMatrix.compute(y_test, y_pred)
    print(f"  Accuracy (programatik): {result['accuracy']:.4f}")
    print(f"  Confusion Matrix: {result['matrix']}")

if __name__ == "__main__":
    main()
