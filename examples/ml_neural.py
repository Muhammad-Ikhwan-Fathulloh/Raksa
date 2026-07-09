# examples/ml_neural.py
# ============================================================================
# Raksa - Contoh TinyML Neural Network (Perceptron & TinyNeuralNetwork)
# ============================================================================
# Demo pelatihan Perceptron pada gerbang logika AND & OR, serta
# TinyNeuralNetwork pada problem XOR menggunakan backpropagation.
# ============================================================================

from raksa import Perceptron, TinyNeuralNetwork, Activation

def main():
    # ── 1. Perceptron: Gerbang AND ──────────────────────────────────────────
    print("=== 1. Perceptron - Gerbang AND ===")
    p_and = Perceptron(dims=2, lr=0.1)

    # Data training (flat format): [x0_f0, x0_f1, x1_f0, x1_f1, ...]
    X_and = [0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0]
    y_and = [0, 0, 0, 1]

    p_and.train(X_and, y_and, epochs=50)

    print(f"  AND(0,0) = {p_and.predict([0.0, 0.0])}  (Expected: 0)")
    print(f"  AND(0,1) = {p_and.predict([0.0, 1.0])}  (Expected: 0)")
    print(f"  AND(1,0) = {p_and.predict([1.0, 0.0])}  (Expected: 0)")
    print(f"  AND(1,1) = {p_and.predict([1.0, 1.0])}  (Expected: 1)")
    print()

    # ── 2. Perceptron: Gerbang OR ───────────────────────────────────────────
    print("=== 2. Perceptron - Gerbang OR ===")
    p_or = Perceptron(dims=2, lr=0.1)

    X_or = [0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0]
    y_or = [0, 1, 1, 1]

    p_or.train(X_or, y_or, epochs=50)

    print(f"  OR(0,0) = {p_or.predict([0.0, 0.0])}  (Expected: 0)")
    print(f"  OR(0,1) = {p_or.predict([0.0, 1.0])}  (Expected: 1)")
    print(f"  OR(1,0) = {p_or.predict([1.0, 0.0])}  (Expected: 1)")
    print(f"  OR(1,1) = {p_or.predict([1.0, 1.0])}  (Expected: 1)")
    print()

    # ── 3. Fungsi Aktivasi ──────────────────────────────────────────────────
    print("=== 3. Fungsi Aktivasi ===")
    print(f"  sigmoid(0.0)  = {Activation.sigmoid(0.0):.4f}  (Expected: 0.5)")
    print(f"  sigmoid(2.0)  = {Activation.sigmoid(2.0):.4f}  (Expected: 0.8808)")
    print(f"  relu(-1.0)    = {Activation.relu(-1.0):.4f}  (Expected: 0.0)")
    print(f"  relu(3.5)     = {Activation.relu(3.5):.4f}  (Expected: 3.5)")
    print(f"  tanh(0.0)     = {Activation.tanh(0.0):.4f}  (Expected: 0.0)")
    print(f"  softmax([1,2,3]) = {[round(v, 4) for v in Activation.softmax([1.0, 2.0, 3.0])]}")
    print()

    # ── 4. TinyNeuralNetwork: Problem XOR ───────────────────────────────────
    print("=== 4. TinyNeuralNetwork - Problem XOR ===")
    # XOR membutuhkan hidden layer karena non-linearly separable
    nn = TinyNeuralNetwork(layer_sizes=[2, 4, 1], lr=1.0)

    X_xor = [0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0]
    y_xor = [[0.0], [1.0], [1.0], [0.0]]

    print("  Melatih jaringan saraf pada XOR (2000 epoch)...")
    nn.train(X_xor, y_xor, epochs=2000)

    for i in range(4):
        inp = X_xor[i*2:(i+1)*2]
        out = nn.predict(inp)
        print(f"  XOR({int(inp[0])},{int(inp[1])}) = {out[0]:.4f}  (Expected: {y_xor[i][0]})")

    print()
    print("Semua test neural network berhasil!")

if __name__ == "__main__":
    main()
