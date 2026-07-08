# verify_client.py
import asyncio
import sys
import json
import math

from raksa import (
    RaksaClient,
    MinMaxScaler,
    StandardScaler,
    PolynomialFeatures,
    LinearForecaster,
    KNN,
    NaiveBayes,
    LogisticRegression,
    DecisionTreeClassifier,
    KMeans
)

async def run_client_test():
    print("[TEST] Memulai pengujian unit Machine Learning Raksa (NocML)...")

    # 1. MinMaxScaler
    print("[TEST] 1. Menguji MinMaxScaler...")
    min_scaler = MinMaxScaler(dims=2, min_vals=[0.0, 10.0], max_vals=[1000.0, 50.0])
    scaled = min_scaler.transform([500.0, 30.0])
    print(f"       Scaled [500.0, 30.0] -> {scaled} (Expected: [0.5, 0.5])")
    assert abs(scaled[0] - 0.5) < 1e-5
    assert abs(scaled[1] - 0.5) < 1e-5

    # 2. StandardScaler
    print("[TEST] 2. Menguji StandardScaler...")
    std_scaler = StandardScaler(dims=2, means=[100.0, 4.0], stddevs=[15.0, 2.0])
    scaled = std_scaler.transform([115.0, 3.0])
    print(f"       Scaled [115.0, 3.0] -> {scaled} (Expected: [1.0, -0.5])")
    assert abs(scaled[0] - 1.0) < 1e-5
    assert abs(scaled[1] - (-0.5)) < 1e-5

    # 3. PolynomialFeatures
    print("[TEST] 3. Menguji PolynomialFeatures (degree 2, 2 inputs)...")
    poly_gen = PolynomialFeatures(degree=2)
    features = poly_gen.transform([2.0, 3.0])
    print(f"       Poly features of [2.0, 3.0] -> {features}")
    # Expected: [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
    expected_poly = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
    for i in range(6):
        assert abs(features[i] - expected_poly[i]) < 1e-5

    # 4. KNN Classifier
    print("[TEST] 4. Menguji KNN Classifier...")
    # Flat format training data
    train_data = [
        0.2, 0.8,
        0.3, 0.7,
        0.8, 0.3,
        0.7, 0.4
    ]
    labels = [1, 1, 0, 0]
    knn = KNN(train_data, labels, num_samples=4, dims=2, k=3)
    prediction = knn.predict([0.25, 0.75])
    print(f"       KNN [0.25, 0.75] -> {prediction} (Expected: 1)")
    assert prediction == 1

    # 5. Naive Bayes Classifier
    print("[TEST] 5. Menguji Naive Bayes...")
    # Mean and Var: [Class0_v0, Class0_v1, Class1_v0, Class1_v1]
    means = [10.0, 5.0, 20.0, 25.0]
    vars = [2.0, 1.0, 3.0, 4.0]
    priors = [0.5, 0.5]
    nb = NaiveBayes(num_classes=2, dims=2, means=means, vars=vars, priors=priors)
    prediction = nb.predict([10.5, 4.8])
    print(f"       NaiveBayes [10.5, 4.8] -> {prediction} (Expected: 0)")
    assert prediction == 0

    # 6. LogisticRegression
    print("[TEST] 6. Menguji Logistic Regression...")
    weights = [2.5, -1.0]
    bias = -3.0
    log_reg = LogisticRegression(dims=2, weights=weights, bias=bias)
    # z = -3.0 + 2.5 * 2.0 - 1.0 * 1.0 = -3.0 + 5.0 - 1.0 = 1.0 -> predict = 1
    # prob = 1 / (1 + exp(-1)) = 0.731
    prediction = log_reg.predict([2.0, 1.0])
    prob = log_reg.predict_proba([2.0, 1.0])
    print(f"       LogReg [2.0, 1.0] -> Pred: {prediction}, Prob: {prob:.4f} (Expected: 1, 0.7311)")
    assert prediction == 1
    assert abs(prob - 0.731058578) < 1e-4

    # 7. DecisionTree Classifier
    print("[TEST] 7. Menguji Decision Tree...")
    # Node structure: feature_idx, threshold, left_child, right_child, value
    # Root: feature 0, threshold 10.0. Go to left node 1, else right node 2
    # Node 1: leaf (value 100)
    # Node 2: leaf (value 200)
    tree_nodes = [
        {"feature_idx": 0, "threshold": 10.0, "left_child": 1, "right_child": 2, "value": -1},
        {"feature_idx": -1, "threshold": 0.0, "left_child": -1, "right_child": -1, "value": 100},
        {"feature_idx": -1, "threshold": 0.0, "left_child": -1, "right_child": -1, "value": 200}
    ]
    tree = DecisionTreeClassifier(tree_nodes)
    pred_low = tree.predict([5.0])
    pred_high = tree.predict([15.0])
    print(f"       Low input [5.0] -> {pred_low} (Expected: 100)")
    print(f"       High input [15.0] -> {pred_high} (Expected: 200)")
    assert pred_low == 100
    assert pred_high == 200

    # 8. K-Means Clustering
    print("[TEST] 8. Menguji K-Means...")
    kmeans = KMeans(k=2, dims=1, centroids=[1.0, 6.0])
    pred = kmeans.predict([5.5])
    print(f"       KMeans predict [5.5] -> Centroid {pred} (Expected: 1)")
    assert pred == 1
    # Run fit test
    dummy_data = [1.1, 1.2, 5.8, 6.0]
    assignments = kmeans.run(dummy_data, num_samples=4, max_iters=5)
    print(f"       KMeans run assignment on {dummy_data} -> {assignments}")
    print(f"       Updated Centroids: {kmeans.centroids}")
    # Samples 1.1, 1.2 close to 1.0 (cluster 0); 5.8, 6.0 close to 6.0 (cluster 1)
    assert assignments == [0, 0, 1, 1]
    assert abs(kmeans.centroids[0] - 1.15) < 1e-3
    assert abs(kmeans.centroids[1] - 5.90) < 1e-3

    # 9. LinearForecaster
    print("[TEST] 9. Menguji LinearForecaster...")
    forecaster = LinearForecaster()
    # Simple line y = 2x + 1
    x_train = [0.0, 1.0, 2.0, 3.0]
    y_train = [1.0, 3.0, 5.0, 7.0]
    forecaster.fit(x_train, y_train)
    nxt = forecaster.forecastNext()
    # next x should be 4.0 -> y should be 9.0
    print(f"       ForecastNext -> {nxt} (Expected: 9.0)")
    assert abs(nxt - 9.0) < 1e-5

    print("[TEST] Semua unit test Machine Learning (NocML) BERHASIL!")
    print("=" * 60)

    # test 10: WebSocket End-to-End
    print("[TEST] 10. Memulai pengujian WebSocket konektivitas server...")
    client = RaksaClient("ws://127.0.0.1:8000/ws/telemetry")
    
    # Test In-situ Native Inference
    weights_mat = [[0.8, -0.4], [0.1, 0.9]]
    biases_mat = [0.1, -0.05]
    model_param = (weights_mat, biases_mat)
    inference_inputs = [1.5, 0.5]
    
    prediction = client.infer(model_param, inference_inputs)
    
    connected = await client.connect()
    if not connected:
        print("[TEST] WARNING: Gagal menghubungkan ke localhost:8000. Melewati pengujian WebSocket.")
        print("[TEST] Verifikasi unit test selesai dengan sukses.")
        return
        
    payload = {
        "model": "NocML_InSitu_V1",
        "features": inference_inputs,
        "prediction": prediction
    }
    
    print("[TEST] Mengirimkan payload telemetry via sync()...")
    sync_success = await client.sync(payload)
    if not sync_success:
        print("[TEST] EROR: Sinkronisasi data gagal.")
        sys.exit(1)
        
    # Read server frame back
    print("[TEST] Membaca respon handshake & sinkronisasi dari server...")
    try:
        response_str = await client.recv()
        print(f"       Respon server diterima: {response_str}")
        res = json.loads(response_str)
        assert res.get("status") == "synchronized"
    except Exception as e:
        print(f"[TEST] Gagal membaca atau memvalidasi respon server: {e}")
        await client.close()
        sys.exit(1)

    await client.close()
    print("[TEST] Seluruh pengujian lokal berhasil diverifikasi secara end-to-end!")

if __name__ == "__main__":
    asyncio.run(run_client_test())
