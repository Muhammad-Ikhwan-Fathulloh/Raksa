# Raksa 🛡️

[English Version](README.md)

**Raksa** (diambil dari bahasa Sansekerta/Indonesia yang berarti *Pelindung* atau *Penjaga*) adalah library MicroPython persistensi-tinggi yang dirancang khusus untuk ekosistem **Noc Lab** dalam workshop **'Tiny Chip, Big Brain'**. Library ini bertindak sebagai jembatan *In-situ Analytics* ujung-ke-ujung (end-to-end) yang mengamankan aliran data telemetry dari perangkat edge (ESP32/RP2040) menuju FastAPI cloud backend secara asinkron.

Dengan performa komputasi native (`@micropython.native`) untuk inferensi model *on-device* dan mekanisme pembersihan memori agresif (`gc.collect()`), Raksa melestarikan stabilitas RAM perangkat mikro Anda dari fragmentasi heap di kala menjalankan beban kerja TinyML secara terus-menerus.

---

## Fitur Utama
- 🔄 **Persistensi WebSocket Mandiri**: Handshake RFC 6455 internal berbasis soket asinkron `uasyncio` tanpa membebani memori dengan library eksternal.
- ⚡ **In-situ Analytics Cepat**: Eksekusi algoritma inferensi TinyML didukung akselerasi `@micropython.native` untuk meminimalkan latensi.
- 🧹 **Konsolidasi Memori Dinamis**: Pemanggilan Garbage Collector terarah di setiap siklus pengiriman mencegah fragmentasi RAM/out-of-memory.
- 🇮🇩 **Arsitektur Nusantara Tangguh**: Konsep penjaga data yang mengedepankan efisiensi, ketangguhan mandiri, dan integrasi erat edge-to-cloud.

---

## Cara Instalasi via `mip`

### 1. Menggunakan `mpremote` (Direkomendasikan)
Hubungkan perangkat Anda ke PC via USB, kemudian jalankan perintah berikut:
```bash
mpremote mip install github:Muhammad-Ikhwan-Fathulloh/Raksa
```

### 2. Memasang langsung via REPL MicroPython
Pastikan board Anda memiliki koneksi Wi-Fi yang aktif, lalu jalankan perintah berikut di REPL:
```python
import mip
mip.install("github:Muhammad-Ikhwan-Fathulloh/Raksa")
```

---

## Contoh Kode Penggunaan Minimalis (ESP32)

Berikut adalah contoh skenario In-situ Analytics & sinkronisasi data ke cloud backend dalam kurang dari 10 baris kode fungsional:

```python
import uasyncio as asyncio
from raksa import RaksaClient

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    model = ([[0.5, -0.2], [0.1, 0.9]], [0.1, 0.0]) # Matriks weight & bias TinyML
    inputs = [0.8, 1.5] # Data sensor mentah
    
    pred = client.infer(model, inputs)
    await client.sync({"model": "NocML_V1", "features": inputs, "prediction": pred})

asyncio.run(main())
```

---

## Contoh Lengkap ESP32

Contoh siap-pakai tersedia di folder [`examples/`](examples/):

| File                                                    | Skenario                                | Sensor                     |
| ------------------------------------------------------- | --------------------------------------- | -------------------------- |
| [`main_edge.py`](examples/main_edge.py)                 | Quick-start minimalis (< 10 baris)      | —                          |
| [`esp32_wifi_sensor.py`](examples/esp32_wifi_sensor.py) | Sensor analog ADC + klasifikasi 3 kelas | LDR / MQ-x / Potensiometer |
| [`esp32_dht_monitor.py`](examples/esp32_dht_monitor.py) | Deteksi anomali suhu & kelembaban       | DHT11 / DHT22              |

### ESP32 + Sensor ADC + Klasifikasi (Cuplikan)

```python
import machine, network, uasyncio as asyncio
from raksa import RaksaClient

# WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("SSID_Anda", "Password_Anda")

# Sensor & Model
adc = machine.ADC(machine.Pin(34))
adc.atten(machine.ADC.ATTN_11DB)
model = ([[0.85, -0.30], [-0.20, 0.75], [0.10, 0.95]], [0.05, -0.10, 0.15])
label = {0: "Normal", 1: "Peringatan", 2: "Bahaya"}

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    await client.connect()
    while True:
        raw = adc.read()
        fitur = [raw / 4095.0, abs(raw / 4095.0 - 0.5)]
        pred = client.infer(model, fitur)
        kls = max(range(len(pred)), key=lambda i: pred[i])
        await client.sync({"kelas": label[kls], "fitur": fitur, "prediksi": pred})
        await asyncio.sleep(5)

asyncio.run(main())
```

### ESP32 + DHT22 Deteksi Anomali (Cuplikan)

```python
import machine, dht, network, uasyncio as asyncio
from raksa import RaksaClient

# WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("SSID_Anda", "Password_Anda")

# Sensor DHT22 & Model Anomali
sensor = dht.DHT22(machine.Pin(4))
model = ([[0.60, -0.35], [-0.45, 0.80]], [0.20, -0.15])

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    await client.connect()
    while True:
        sensor.measure()
        suhu, hum = sensor.temperature(), sensor.humidity()
        fitur = [(suhu - 15) / 35, (hum - 20) / 75]  # Normalisasi
        pred = client.infer(model, fitur)
        status = "ANOMALI" if pred[1] > pred[0] else "NORMAL"
        await client.sync({"suhu": suhu, "kelembaban": hum, "status": status, "prediksi": pred})
        await asyncio.sleep(10)

asyncio.run(main())
```

---

## Lisensi & Hak Cipta
Dikembangkan oleh **Noc Lab** untuk mendukung edukasi perangkat keras TinyML di Indonesia. Dilindungi oleh Lisensi MIT.