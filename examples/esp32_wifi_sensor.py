# examples/esp32_wifi_sensor.py
# ============================================================================
# Raksa - Contoh Lengkap ESP32: WiFi + Sensor ADC + Inferensi TinyML + Sync
# ============================================================================
# Skenario: Membaca data analog dari sensor (misalnya LDR, potensiometer,
# atau sensor gas MQ-x) melalui pin ADC ESP32, menjalankan inferensi TinyML
# secara lokal, kemudian mengirimkan hasil ke cloud backend via WebSocket.
#
# Hardware:
#   - ESP32 DevKit V1 / ESP32-S3
#   - Sensor analog terhubung ke GPIO34 (ADC1_CH6)
#   - LED indikator di GPIO2 (built-in LED)
#
# Instalasi library:
#   mpremote mip install github:Muhammad-Ikhwan-Fathulloh/Raksa
# ============================================================================

import machine
import network
import utime
import uasyncio as asyncio
import gc
from raksa import RaksaClient

# ── Konfigurasi ──────────────────────────────────────────────────────────────

WIFI_SSID     = "NocLab_WiFi"
WIFI_PASSWORD = "workshop2026"

# Ganti dengan IP komputer server FastAPI Anda
BACKEND_URL   = "ws://192.168.1.100:8000/ws/telemetry"

# Pin hardware
ADC_PIN       = 34   # GPIO34 — Input Only, cocok untuk ADC
LED_PIN       = 2    # GPIO2  — Built-in LED (indikator status)

# Interval pengiriman data (detik)
SYNC_INTERVAL = 5

# Model TinyML sederhana — Dense Layer (2 input → 3 output) + ReLU
# Contoh: klasifikasi kondisi sensor menjadi 3 kelas:
#   [0] = Normal, [1] = Peringatan, [2] = Bahaya
MODEL_WEIGHTS = [
    [0.85, -0.30],   # Neuron 0 (Normal)
    [-0.20,  0.75],  # Neuron 1 (Peringatan)
    [0.10,   0.95],  # Neuron 2 (Bahaya)
]
MODEL_BIASES  = [0.05, -0.10, 0.15]
MODEL         = (MODEL_WEIGHTS, MODEL_BIASES)

LABEL_MAP     = {0: "Normal", 1: "Peringatan", 2: "Bahaya"}

# ── Fungsi WiFi ──────────────────────────────────────────────────────────────

def connect_wifi(ssid: str, password: str) -> str:
    """Menghubungkan ESP32 ke jaringan WiFi dan mengembalikan alamat IP."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        print(f"[WIFI] Sudah terhubung: {wlan.ifconfig()[0]}")
        return wlan.ifconfig()[0]

    print(f"[WIFI] Menghubungkan ke '{ssid}'...")
    wlan.connect(ssid, password)

    timeout = 20  # detik
    while not wlan.isconnected() and timeout > 0:
        utime.sleep(1)
        timeout -= 1
        print(f"[WIFI] Menunggu koneksi... ({20 - timeout}s)")

    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"[WIFI] Terhubung! IP: {ip}")
        return ip
    else:
        raise RuntimeError("[WIFI] Gagal terhubung ke WiFi. Periksa SSID/Password.")

# ── Fungsi Sensor ────────────────────────────────────────────────────────────

def read_sensor(adc: machine.ADC) -> list:
    """
    Membaca nilai sensor ADC dan menormalisasi ke rentang 0.0 - 1.0.
    Mengembalikan feature vector [normalized_value, delta_from_midpoint].
    """
    raw = adc.read()             # 0 - 4095 (12-bit ADC)
    normalized = raw / 4095.0    # Normalisasi ke 0.0 - 1.0
    delta = abs(normalized - 0.5)  # Jarak dari titik tengah (fitur kedua)
    return [round(normalized, 4), round(delta, 4)]

# ── Fungsi Klasifikasi ───────────────────────────────────────────────────────

def classify(prediction: list) -> tuple:
    """Mengambil indeks kelas tertinggi (argmax) dari hasil inferensi."""
    max_val = prediction[0]
    max_idx = 0
    for i in range(1, len(prediction)):
        if prediction[i] > max_val:
            max_val = prediction[i]
            max_idx = i
    return max_idx, LABEL_MAP.get(max_idx, "Unknown")

# ── Loop Utama ───────────────────────────────────────────────────────────────

async def main():
    # 1. Koneksi WiFi
    ip_address = connect_wifi(WIFI_SSID, WIFI_PASSWORD)
    gc.collect()

    # 2. Inisialisasi hardware
    led = machine.Pin(LED_PIN, machine.Pin.OUT)
    adc = machine.ADC(machine.Pin(ADC_PIN))
    adc.atten(machine.ADC.ATTN_11DB)   # Rentang input penuh: 0 - 3.3V
    adc.width(machine.ADC.WIDTH_12BIT) # Resolusi 12-bit

    # 3. Inisialisasi RaksaClient
    client = RaksaClient(BACKEND_URL)
    connected = await client.connect()

    if not connected:
        print("[MAIN] Gagal terhubung ke backend. Memulai ulang dalam 10 detik...")
        utime.sleep(10)
        machine.reset()

    print("[MAIN] Sistem siap. Memulai loop akuisisi data & inferensi...")
    print(f"[MAIN] Interval sinkronisasi: setiap {SYNC_INTERVAL} detik")
    print("=" * 60)

    cycle = 0

    # 4. Loop akuisisi & sinkronisasi berkelanjutan
    while True:
        cycle += 1
        led.on()  # Indikator: sedang memproses

        # Baca sensor
        features = read_sensor(adc)

        # Inferensi lokal dengan @micropython.native
        prediction = client.infer(MODEL, features)
        class_idx, class_label = classify(prediction)

        # Cetak ke serial monitor
        print(f"[#{cycle:04d}] Sensor: {features} | "
              f"Prediksi: {prediction} | "
              f"Kelas: {class_label} ({class_idx})")

        # Kirim ke cloud backend
        payload = {
            "device": "ESP32",
            "ip": ip_address,
            "cycle": cycle,
            "model": "NocML_Sensor_V1",
            "features": features,
            "prediction": prediction,
            "class": class_label
        }

        success = await client.sync(payload)

        if success:
            led.off()  # Indikator: berhasil dikirim
        else:
            # Blink cepat sebagai tanda gagal
            for _ in range(3):
                led.on()
                utime.sleep_ms(100)
                led.off()
                utime.sleep_ms(100)
            # Coba reconnect
            print("[MAIN] Koneksi terputus. Mencoba menyambung ulang...")
            await client.connect()

        # Tunggu interval berikutnya
        await asyncio.sleep(SYNC_INTERVAL)

# Entry Point
asyncio.run(main())
