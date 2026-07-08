# examples/esp32_dht_monitor.py
# ============================================================================
# Raksa - Contoh Lengkap ESP32: Monitoring Suhu & Kelembaban (DHT11/DHT22)
# ============================================================================
# Skenario: Membaca data suhu dan kelembaban dari sensor DHT, menjalankan
# inferensi anomaly detection secara lokal menggunakan model TinyML,
# kemudian mengirimkan telemetry ke cloud backend via WebSocket.
#
# Hardware:
#   - ESP32 DevKit V1 / ESP32-S3
#   - DHT11 atau DHT22 terhubung ke GPIO4
#   - LED indikator di GPIO2 (built-in LED)
#
# Wiring DHT:
#   VCC → 3.3V, GND → GND, DATA → GPIO4 (pull-up 10kΩ ke 3.3V)
#
# Instalasi library:
#   mpremote mip install github:Muhammad-Ikhwan-Fathulloh/Raksa
# ============================================================================

import machine
import dht
import network
import utime
import uasyncio as asyncio
import gc
from raksa import RaksaClient

# ── Konfigurasi ──────────────────────────────────────────────────────────────

WIFI_SSID     = "NocLab_WiFi"
WIFI_PASSWORD = "workshop2026"
BACKEND_URL   = "ws://192.168.1.100:8000/ws/telemetry"

DHT_PIN       = 4     # GPIO4  — Data pin DHT
LED_PIN       = 2     # GPIO2  — Built-in LED
DHT_TYPE      = "DHT22"  # Pilih: "DHT11" atau "DHT22"

SYNC_INTERVAL = 10    # Interval pengiriman (detik)
DEVICE_ID     = "ESP32-DHT-01"

# Model Anomaly Detection TinyML
# Input: [suhu_normalized, kelembaban_normalized]
# Output: [skor_normal, skor_anomali]
# Jika skor_anomali > skor_normal → kondisi tidak wajar terdeteksi
MODEL_WEIGHTS = [
    [ 0.60, -0.35],  # Neuron Normal
    [-0.45,  0.80],  # Neuron Anomali
]
MODEL_BIASES  = [0.20, -0.15]
MODEL         = (MODEL_WEIGHTS, MODEL_BIASES)

# Rentang suhu & kelembaban untuk normalisasi
TEMP_MIN, TEMP_MAX = 15.0, 50.0  # Celcius
HUM_MIN,  HUM_MAX = 20.0, 95.0  # Persen

# ── Fungsi WiFi ──────────────────────────────────────────────────────────────

def connect_wifi(ssid: str, password: str) -> str:
    """Menghubungkan ESP32 ke jaringan WiFi."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"[WIFI] Sudah terhubung: {ip}")
        return ip

    print(f"[WIFI] Menghubungkan ke '{ssid}'...")
    wlan.connect(ssid, password)

    for i in range(20):
        if wlan.isconnected():
            break
        utime.sleep(1)
        print(f"[WIFI] Menunggu... ({i + 1}s)")

    if not wlan.isconnected():
        raise RuntimeError("[WIFI] Timeout! Periksa kredensial WiFi.")

    ip = wlan.ifconfig()[0]
    print(f"[WIFI] Terhubung! IP: {ip}")
    return ip

# ── Fungsi Sensor DHT ────────────────────────────────────────────────────────

def init_dht(pin: int, dht_type: str):
    """Inisialisasi sensor DHT11 atau DHT22."""
    p = machine.Pin(pin)
    if dht_type == "DHT22":
        return dht.DHT22(p)
    else:
        return dht.DHT11(p)

def read_dht(sensor) -> dict:
    """Membaca suhu dan kelembaban, mengembalikan dict data mentah."""
    sensor.measure()
    temp = sensor.temperature()  # Celcius
    hum  = sensor.humidity()     # Persen
    return {"temperature": temp, "humidity": hum}

def normalize_features(temp: float, hum: float) -> list:
    """Normalisasi suhu dan kelembaban ke rentang 0.0 - 1.0 untuk inferensi."""
    t_norm = max(0.0, min(1.0, (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)))
    h_norm = max(0.0, min(1.0, (hum - HUM_MIN) / (HUM_MAX - HUM_MIN)))
    return [round(t_norm, 4), round(h_norm, 4)]

# ── Loop Utama ───────────────────────────────────────────────────────────────

async def main():
    # 1. Koneksi WiFi
    ip_address = connect_wifi(WIFI_SSID, WIFI_PASSWORD)
    gc.collect()

    # 2. Inisialisasi hardware
    led = machine.Pin(LED_PIN, machine.Pin.OUT)
    sensor = init_dht(DHT_PIN, DHT_TYPE)
    print(f"[MAIN] Sensor {DHT_TYPE} diinisialisasi pada GPIO{DHT_PIN}")

    # 3. Inisialisasi RaksaClient
    client = RaksaClient(BACKEND_URL)
    connected = await client.connect()

    if not connected:
        print("[MAIN] Gagal terhubung ke backend. Reset dalam 10 detik...")
        utime.sleep(10)
        machine.reset()

    print(f"[MAIN] Device: {DEVICE_ID}")
    print(f"[MAIN] Interval sync: {SYNC_INTERVAL}s")
    print("=" * 60)

    cycle = 0
    
    # 4. Buffer untuk rata-rata bergerak (moving average) 5 pembacaan terakhir
    temp_buffer = []
    hum_buffer = []
    BUFFER_SIZE = 5

    while True:
        cycle += 1
        led.on()

        try:
            # Baca sensor DHT
            reading = read_dht(sensor)
            temp = reading["temperature"]
            hum  = reading["humidity"]

            # Tambahkan ke buffer & hitung rata-rata
            temp_buffer.append(temp)
            hum_buffer.append(hum)
            if len(temp_buffer) > BUFFER_SIZE:
                temp_buffer.pop(0)
            if len(hum_buffer) > BUFFER_SIZE:
                hum_buffer.pop(0)

            avg_temp = sum(temp_buffer) / len(temp_buffer)
            avg_hum  = sum(hum_buffer) / len(hum_buffer)

            # Normalisasi untuk input model
            features = normalize_features(avg_temp, avg_hum)

            # Inferensi anomaly detection lokal
            prediction = client.infer(MODEL, features)
            is_anomaly = prediction[1] > prediction[0]
            status = "ANOMALI" if is_anomaly else "NORMAL"

            # Tampilkan di serial monitor
            print(f"[#{cycle:04d}] "
                  f"Suhu: {temp:.1f}°C (avg: {avg_temp:.1f}) | "
                  f"Hum: {hum:.0f}% (avg: {avg_hum:.0f}) | "
                  f"Status: {status}")

            # Kirim telemetry ke cloud
            payload = {
                "device_id": DEVICE_ID,
                "cycle": cycle,
                "model": "NocML_AnomalyDet_V1",
                "raw": {"temperature": temp, "humidity": hum},
                "averaged": {"temperature": round(avg_temp, 2), "humidity": round(avg_hum, 1)},
                "features": features,
                "prediction": prediction,
                "status": status,
                "is_anomaly": is_anomaly
            }

            success = await client.sync(payload)
            led.off()

            if not success:
                print("[MAIN] Sync gagal, mencoba reconnect...")
                await client.connect()

            # Jika anomali terdeteksi, blink LED sebagai peringatan
            if is_anomaly:
                for _ in range(5):
                    led.on()
                    utime.sleep_ms(150)
                    led.off()
                    utime.sleep_ms(150)

        except OSError as e:
            print(f"[MAIN] Kesalahan pembacaan sensor: {e}")
            led.off()

        # Bersihkan memori di setiap siklus
        gc.collect()
        await asyncio.sleep(SYNC_INTERVAL)

# Entry Point
asyncio.run(main())
