# examples/ml_clustering.py
# ============================================================================
# Raksa - Contoh Clustering TinyML (K-Means In-situ Training & Prediction)
# ============================================================================
# Demo pengelompokan data sensor getaran motor secara dinamis (in-situ fit)
# untuk mendeteksi deviasi/kelompok anomali secara mandiri di perangkat.
# ============================================================================

from raksa import KMeans

def main():
    # ── K-Means Anomaly Grouping ─────────────────────────────────────────────
    print("=== K-Means Clustering (Prediksi & In-situ Training) ===")
    
    # 1. Menggunakan Centroids yang Sudah Dilatih Sebelumnya (Sklearn export)
    # K=2 klaster (Normal vs Abnormal), dimensi data getaran 1D
    # Kita inisialisasi centroid awal: Normal = 1.1, Abnormal = 6.0
    initial_centroids = [1.1, 6.0]
    kmeans = KMeans(k=2, dims=1, centroids=initial_centroids)
    
    # Prediksi getaran baru (misal terdeteksi 5.7 - getaran tinggi)
    new_vibration = [5.7]
    cluster_idx = kmeans.predict(new_vibration)
    centroid_val = kmeans.centroids[cluster_idx]
    
    print(f"Prediksi getaran lama {new_vibration}:")
    print(f"  - Masuk Klaster Centroid Indeks: {cluster_idx} (Lokasi Centroid: {centroid_val:.2f})")
    print(f"  - Status Getaran: {'ABNORMAL/MUTASI' if centroid_val > 3.0 else 'NORMAL'}\n")


    # 2. In-situ Training (Fitting On-device tanpa dependensi eksternal)
    # Jika kita punya aliran data tak berlabel baru dan ingin memutakhirkan model kita:
    print("Melakukan latihan in-situ (Fitting data sensor getaran baru)...")
    vibration_stream = [1.0, 1.2, 0.9, 5.5, 6.2, 5.9] # Aliran sensor getaran
    
    # Inisalasi penyimpan klaster per sampel getaran
    assignments = [0] * len(vibration_stream)
    
    # Latih model K-means lokal selama maksimum 10 iterasi
    kmeans.run(vibration_stream, num_samples=6, assignments=assignments, max_iters=10)
    
    print(f"Hasil Fit Klaster Data : {vibration_stream}")
    print(f"Pembagian Klaster Sampel: {assignments}")
    
    # Cetak pembaruan lokasi centroid baru setelah penyesuaian data getaran
    print("\nLokasi Centroid Yang Diperbarui:")
    for idx in range(kmeans.k):
        c_val = kmeans.centroids[idx]
        print(f"  - Centroid [{idx}]: {c_val:.4f} (Kelompok: {'Abnormal' if c_val > 3.0 else 'Normal'})")

if __name__ == "__main__":
    main()
