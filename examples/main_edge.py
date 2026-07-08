# examples/main_edge.py
# ============================================================================
# Raksa - Contoh Minimalis (< 10 baris fungsional)
# ============================================================================
# Gunakan file ini sebagai titik awal cepat untuk memahami alur kerja Raksa.
# Untuk contoh lengkap dengan WiFi, sensor, dan loop, lihat:
#   - esp32_wifi_sensor.py   → Sensor ADC + Klasifikasi TinyML
#   - esp32_dht_monitor.py   → DHT11/DHT22 + Anomaly Detection
# ============================================================================

try:
    import uasyncio as asyncio
except ImportError:
    import asyncio
from raksa import RaksaClient

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    model = ([[0.5, -0.2], [0.1, 0.9]], [0.1, 0.0])  # TinyML weights & biases
    inputs = [0.8, 1.5]  # Sensor feature vector

    prediction = client.infer(model, inputs)
    await client.sync({"model": "NocML_V1", "features": inputs, "prediction": prediction})

asyncio.run(main())
