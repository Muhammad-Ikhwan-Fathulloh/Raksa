# examples/ml_forecasting.py
# ============================================================================
# Raksa - Contoh Peramalan TinyML (Simple Linear Forecaster)
# ============================================================================
# Skenario peramalan trend masa depan (misalnya tren suhu ruangan menit depan)
# menggunakan model fit regresi linear sederhana langsung di perangkat edge.
# ============================================================================

from raksa import LinearForecaster

def main():
    # ── Linear Forecaster (Simple Linear Regression) ─────────────────────────
    print("=== LinearForecaster: Peramalan Suhu AC Proaktif ===")
    
    # Data latihan: 5 indeks waktu menit vs Suhu (°C) yang tercatat
    time_indices = [0.0, 1.0, 2.0, 3.0, 4.0]
    temp_records = [24.0, 24.5, 25.1, 25.4, 26.0]  # Menunjukkan tren kenaikan suhu
    
    # Inisialisasi model
    forecaster = LinearForecaster()
    
    # Latih regresi linear (fit m & c) langsung di device
    # y = m * x + c
    forecaster.fit(time_indices, temp_records)
    
    print("Parameter Regresi Hasil Fitting:")
    print(f"  - Gradien Tren (Slope m)  : {forecaster.m:.4f} °C per menit")
    print(f"  - Intersept Konstanta (c): {forecaster.c:.4f} °C")
    
    # Prediksi/Forecasting untuk menit berikutnya (menit ke-5)
    predicted_temp = forecaster.forecastNext()
    print(f"\nHasil Estimasi Suhu Menit ke-5: {predicted_temp:.2f} °C")
    
    # Evaluasi nilai spesifik (Menit ke-10)
    future_time = 10.0
    future_temp = forecaster.predict(future_time)
    print(f"Hasil Estimasi Suhu Menit ke-10: {future_temp:.2f} °C")
    
    # Keputusan kontrol cerdas berbasis peramalan (proaktif)
    if predicted_temp > 26.5:
        print("\n[ALERT] Prediksi suhu melampaui batas kenyamanan!")
        print(" -> Tindakan Proaktif: Menyalakan pendingin udara (AC) lebih awal.")

if __name__ == "__main__":
    main()
