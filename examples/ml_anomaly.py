# examples/ml_anomaly.py
# ============================================================================
# Raksa - Contoh Deteksi Anomali & PCA (AnomalyDetector, PCA)
# ============================================================================
# Demo deteksi anomali berbasis Z-Score pada data sensor, dan reduksi
# dimensionalitas menggunakan PCA untuk visualisasi data edge.
# ============================================================================

from raksa import AnomalyDetector, PCA

def main():
    # ── 1. AnomalyDetector (Z-Score) ────────────────────────────────────────
    print("=== 1. AnomalyDetector - Deteksi Anomali Sensor ===")

    # Data sensor normal: [suhu, kelembaban] (flat format)
    sensor_data = [
        25.0, 60.0,   # normal
        26.0, 58.0,   # normal
        24.5, 62.0,   # normal
        25.5, 59.0,   # normal
        25.0, 61.0,   # normal
        24.0, 63.0,   # normal
        26.5, 57.0,   # normal
        25.0, 60.0,   # normal
    ]

    detector = AnomalyDetector(threshold=2.0)
    detector.fit(sensor_data, n_samples=8, dims=2)

    print(f"  Mean suhu: {detector.means[0]:.2f}, Mean kelembaban: {detector.means[1]:.2f}")
    print(f"  Std suhu:  {detector.stds[0]:.2f}, Std kelembaban:  {detector.stds[1]:.2f}")
    print()

    # Test data: campuran normal dan anomali
    test_samples = [
        ([25.0, 60.0], "Normal"),
        ([26.0, 59.0], "Normal"),
        ([35.0, 60.0], "Anomali (suhu tinggi)"),
        ([25.0, 90.0], "Anomali (kelembaban tinggi)"),
        ([10.0, 20.0], "Anomali (semua rendah)"),
    ]

    for sample, label in test_samples:
        z_score = detector.score(sample)
        is_anomaly = detector.detect(sample)
        status = "ANOMALI" if is_anomaly else "NORMAL"
        print(f"  {sample} -> Z={z_score:.2f} [{status}]  ({label})")

    print()

    # ── 2. PCA - Reduksi Dimensi ────────────────────────────────────────────
    print("=== 2. PCA - Reduksi Dimensionalitas ===")

    # Data 3 dimensi: [fitur_1, fitur_2, fitur_3] (flat format)
    data_3d = [
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        3.0, 6.0, 9.0,
        4.0, 8.0, 12.0,
        5.0, 10.0, 15.0,
        1.5, 3.0, 4.5,
        2.5, 5.0, 7.5,
        3.5, 7.0, 10.5,
    ]

    pca = PCA(n_components=2)
    pca.fit(data_3d, n_samples=8, dims=3)

    print(f"  Komponen utama 1: {[round(v, 4) for v in pca.components[0]]}")
    print(f"  Komponen utama 2: {[round(v, 4) for v in pca.components[1]]}")
    print()

    # Transformasi beberapa sampel
    print("  Hasil reduksi 3D -> 2D:")
    for i in range(4):
        sample = data_3d[i*3:(i+1)*3]
        reduced = pca.transform(sample)
        print(f"    {sample} -> [{reduced[0]:.4f}, {reduced[1]:.4f}]")

    print()

    # ── 3. PCA + AnomalyDetector Gabungan ───────────────────────────────────
    print("=== 3. PCA + AnomalyDetector (Pipeline) ===")

    # Reduksi 3D ke 1D lalu deteksi anomali
    pca_1d = PCA(n_components=1)
    pca_1d.fit(data_3d, n_samples=8, dims=3)

    # Transformasi semua data ke 1D
    reduced_flat = []
    for i in range(8):
        sample = data_3d[i*3:(i+1)*3]
        r = pca_1d.transform(sample)
        reduced_flat.append(r[0])

    # Fit anomaly detector pada data 1D
    det_1d = AnomalyDetector(threshold=2.0)
    det_1d.fit(reduced_flat, n_samples=8, dims=1)

    # Test: data normal vs outlier
    normal_3d = [3.0, 6.0, 9.0]
    outlier_3d = [20.0, 1.0, 50.0]

    r_normal = pca_1d.transform(normal_3d)
    r_outlier = pca_1d.transform(outlier_3d)

    print(f"  Normal  {normal_3d} -> PCA: {r_normal[0]:.4f} -> Anomali: {det_1d.detect(r_normal)}")
    print(f"  Outlier {outlier_3d} -> PCA: {r_outlier[0]:.4f} -> Anomali: {det_1d.detect(r_outlier)}")

    print()
    print("Semua test anomali & PCA berhasil!")

if __name__ == "__main__":
    main()
