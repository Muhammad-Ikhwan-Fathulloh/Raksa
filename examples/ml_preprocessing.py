# examples/ml_preprocessing.py
# ============================================================================
# Raksa - Contoh Preprocessing TinyML (MinMaxScaler, StandardScaler, Polynomials)
# ============================================================================
# Demo preprocessing untuk menormalisasi, menstandarkan, dan memperluas dimensi
# fitur sensor untuk input model machine learning.
# ============================================================================

from raksa import MinMaxScaler, StandardScaler, PolynomialFeatures

def main():
    # ── 1. MinMaxScaler ──────────────────────────────────────────────────────
    print("=== 1. MinMaxScaler ===")
    # Skala fitur sensor dari rentang mentah [0-1023] & [10-50] ke [0.0 - 1.0]
    min_vals = [0.0, 10.0]
    max_vals = [1023.0, 50.0]
    scaler_minmax = MinMaxScaler(dims=2, min_vals=min_vals, max_vals=max_vals)
    
    # Transformasi data sensor mentah
    raw_sensor = [300.0, 38.0]
    scaled_minmax = scaler_minmax.transform(raw_sensor)
    print(f"Sensor mentah: {raw_sensor}")
    print(f"Hasil MinMax Scale: {scaled_minmax}")
    
    # Optimasi RAM (In-situ): Melakukan scaling langsung ke list penampung tanpa alokasi list baru
    output_buffer = [0.0, 0.0]
    scaler_minmax.transform(raw_sensor, out=output_buffer)
    print(f"In-situ Buffer Scale: {output_buffer}\n")


    # ── 2. StandardScaler ────────────────────────────────────────────────────
    print("=== 2. StandardScaler ===")
    # Standardisasi berdasarkan mean & stddev populasi
    means = [512.0, 25.0]
    stddevs = [150.0, 5.0]
    scaler_std = StandardScaler(dims=2, means=means, stddevs=stddevs)
    
    scaled_std = scaler_std.transform(raw_sensor)
    print(f"Sensor mentah: {raw_sensor}")
    print(f"Hasil Standard Scale: {scaled_std}\n")


    # ── 3. PolynomialFeatures ────────────────────────────────────────────────
    print("=== 3. PolynomialFeatures ===")
    # Menghasilkan polinomial orde 2 dari input [x, y], menghasilkan fitur:
    # [1, x, y, x^2, x*y, y^2]
    poly_gen = PolynomialFeatures(degree=2)
    
    input_fitur = [2.0, 3.0]
    poly_features = poly_gen.transform(input_fitur)
    print(f"Fitur Asal: {input_fitur}")
    print(f"Fitur Polinomial Orde 2: {poly_features}")
    # Indeks fitur polinomial yang terbentuk:
    # [0]=1.0 (intercept), [1]=2.0 (x), [2]=3.0 (y), [3]=4.0 (x^2), [4]=6.0 (x*y), [5]=9.0 (y^2)

if __name__ == "__main__":
    main()
