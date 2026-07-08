# verify_client.py
import asyncio
import sys
import json
from raksa import RaksaClient

async def run_client_test():
    print("[TEST] Memulai pengujian lokal RaksaClient...")
    client = RaksaClient("ws://127.0.0.1:8000/ws/telemetry")
    
    # Test 1: In-situ Native Inference
    print("[TEST] 1. Menguji fungsi inferensi In-situ (@micropython.native)...")
    # Weights for a model with 2 inputs, 2 outputs
    weights = [[0.8, -0.4], [0.1, 0.9]]
    biases = [0.1, -0.05]
    model = (weights, biases)
    inputs = [1.5, 0.5]
    
    prediction = client.infer(model, inputs)
    expected = [
        max(0.0, 1.5 * 0.8 + 0.5 * -0.4 + 0.1),
        max(0.0, 1.5 * 0.1 + 0.5 * 0.9 - 0.05)
    ]
    print(f"       Inputs: {inputs}")
    print(f"       Output hasil inferensi: {prediction} (Expected: {expected})")
    assert abs(prediction[0] - expected[0]) < 1e-5, f"Mismatch index 0. Got {prediction[0]}, expected {expected[0]}"
    assert abs(prediction[1] - expected[1]) < 1e-5, f"Mismatch index 1. Got {prediction[1]}, expected {expected[1]}"
    print("       -> Uji inferensi native sukses.")

    # Test 2: Connecting and Synchronizing
    print("[TEST] 2. Menguji konektivitas WebSocket ke mock server...")
    connected = await client.connect()
    if not connected:
        print("[TEST] EROR: Gagal menghubungkan ke localhost:8000. Pastikan FastAPI backend sudah berjalan.")
        sys.exit(1)
        
    payload = {
        "model": "NocML_InSitu_V1",
        "features": inputs,
        "prediction": prediction
    }
    
    print("[TEST] 3. Mengirimkan payload telemetry via sync()...")
    sync_success = await client.sync(payload)
    if not sync_success:
        print("[TEST] EROR: Sinkronisasi data gagal.")
        sys.exit(1)
    print("       -> Pengiriman telemetry sukses.")

    # Read server frame back
    print("[TEST] 4. Membaca respon handshake & sinkronisasi dari server...")
    try:
        response_str = await client.recv()
        print(f"       Respon server diterima: {response_str}")
        
        res = json.loads(response_str)
        assert res.get("status") == "synchronized", f"Status mismatch: {res.get('status')}"
        print("       -> Respon server tervalidasi dengan baik.")
    except Exception as e:
        print(f"[TEST] Gagal membaca atau memvalidasi respon server: {e}")
        import traceback
        traceback.print_exc()
        await client.close()
        sys.exit(1)

    await client.close()
    print("[TEST] Seluruh pengujian lokal berhasil diverifikasi secara end-to-end!")

if __name__ == "__main__":
    asyncio.run(run_client_test())
