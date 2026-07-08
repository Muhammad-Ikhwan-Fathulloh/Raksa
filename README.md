# Raksa 🛡️

**Raksa** (derived from Sanskrit/Indonesian word for *Protector* or *Guardian*) is a high-persistence, self-contained MicroPython library built for the **Noc Lab** TinyML ecosystem as part of the **'Tiny Chip, Big Brain'** workshop. It serves as an end-to-end *In-situ Analytics* gateway, securing stable asynchronous telemetry data stream from edge devices (ESP32/RP2040) to a FastAPI cloud backend.

Leveraging native compiler execution (`@micropython.native`) for fast on-device inference and memory consolidations (`gc.collect()`), Raksa protects MicroPython devices from heap fragmentation during high-velocity TinyML operations.

---

## Key Features
- 🔄 **Persistent WebSockets**: Pure python RFC 6455 client implementation utilizing raw asynchronous `uasyncio` streams without external footprint overhead.
- ⚡ **High-Speed In-situ Analytics**: Edge inference calculations optimized directly via `@micropython.native` decorators to ensure minimal loop latency.
- 🧹 **Heap Consolidation**: Garbage collection executed dynamically inside data sync cycles to prevent RAM fragmentation on constrained hardware.
- 🇮🇩 **Nusantara Resiliency**: Promotes self-reliance, stability, and secure data guardianship for resource-constrained edge-to-cloud architectures.

---

## Installation via `mip`

### 1. Using `mpremote` (Recommended)
Connect your microcontroller board to your pc via USB/Serial and run:
```bash
mpremote mip install github:Muhammad-Ikhwan-Fathulloh/Raksa
```

### 2. Directly in MicroPython REPL
Ensure your microcontroller board has an active Wi-Fi connection, then run:
```python
import mip
mip.install("github:Muhammad-Ikhwan-Fathulloh/Raksa")
```

---

## Minimalist Code Example (ESP32 / RP2040)

Below is an end-to-end usage code highlighting edge inference and data synchronization under 10 lines of functional code:

```python
import uasyncio as asyncio
from raksa import RaksaClient

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    model = ([[0.5, -0.2], [0.1, 0.9]], [0.1, 0.0]) # TinyML weights & biases
    inputs = [0.8, 1.5] # Raw sensor variables
    
    pred = client.infer(model, inputs)
    await client.sync({"model": "NocML_V1", "features": inputs, "prediction": pred})

asyncio.run(main())
```

---

## Complete ESP32 Examples

Full production-ready examples are available under the [`examples/`](examples/) folder:

| File                                                    | Scenario                                   | Sensors                    |
| ------------------------------------------------------- | ------------------------------------------ | -------------------------- |
| [`main_edge.py`](examples/main_edge.py)                 | Quick-start minimalist (< 10 lines)        | —                          |
| [`esp32_wifi_sensor.py`](examples/esp32_wifi_sensor.py) | ADC analog sensor + 3-class classification | LDR / MQ-x / Potentiometer |
| [`esp32_dht_monitor.py`](examples/esp32_dht_monitor.py) | Temperature & humidity anomaly detection   | DHT11 / DHT22              |

### ESP32 + ADC Sensor Classification (Snippet)

```python
import machine, network, uasyncio as asyncio
from raksa import RaksaClient

# WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("YourSSID", "YourPassword")

# Sensor & Model
adc = machine.ADC(machine.Pin(34))
adc.atten(machine.ADC.ATTN_11DB)
model = ([[0.85, -0.30], [-0.20, 0.75], [0.10, 0.95]], [0.05, -0.10, 0.15])
labels = {0: "Normal", 1: "Warning", 2: "Danger"}

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    await client.connect()
    while True:
        raw = adc.read()
        features = [raw / 4095.0, abs(raw / 4095.0 - 0.5)]
        pred = client.infer(model, features)
        cls = max(range(len(pred)), key=lambda i: pred[i])
        await client.sync({"class": labels[cls], "features": features, "prediction": pred})
        await asyncio.sleep(5)

asyncio.run(main())
```

### ESP32 + DHT22 Anomaly Detection (Snippet)

```python
import machine, dht, network, uasyncio as asyncio
from raksa import RaksaClient

# WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("YourSSID", "YourPassword")

# DHT22 Sensor & Anomaly Model
sensor = dht.DHT22(machine.Pin(4))
model = ([[0.60, -0.35], [-0.45, 0.80]], [0.20, -0.15])

async def main():
    client = RaksaClient("ws://192.168.1.100:8000/ws/telemetry")
    await client.connect()
    while True:
        sensor.measure()
        temp, hum = sensor.temperature(), sensor.humidity()
        features = [(temp - 15) / 35, (hum - 20) / 75]  # Normalize
        pred = client.infer(model, features)
        status = "ANOMALY" if pred[1] > pred[0] else "NORMAL"
        await client.sync({"temp": temp, "hum": hum, "status": status, "prediction": pred})
        await asyncio.sleep(10)

asyncio.run(main())
```

---

## Licenses & Copyrights
Developed by **Noc Lab** for nurturing TinyML literacy on hardware developers. Licensed under the MIT License.