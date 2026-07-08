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

## Licenses & Copyrights
Developed by **Noc Lab** for nurturing TinyML literacy on hardware developers. Licensed under the MIT License.