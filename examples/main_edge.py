# examples/main_edge.py
try:
    import uasyncio as asyncio
except ImportError:
    import asyncio
from raksa import RaksaClient

async def main():
    client = RaksaClient("ws://127.0.0.1:8000/ws/telemetry")
    model = ([[0.5, -0.2], [0.1, 0.9]], [0.1, 0.0]) # TinyML weights & biases
    inputs = [0.8, 1.5]
    
    prediction = client.infer(model, inputs)
    await client.sync({"model": "NocML_V1", "features": inputs, "prediction": prediction})

asyncio.run(main())
