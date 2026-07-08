from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import json

app = FastAPI(title="Raksa In-situ Analytics Cloud Backend - Noc Lab")

@app.get("/")
def read_root():
    return {"status": "online", "application": "Noc Lab Edge Cloud Gateway"}

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[BACKEND] Kamera/Sensor Edge (RaksaClient) terhubung.")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"[BACKEND] Data diterima: {data}")
            try:
                payload = json.loads(data)
                print(f"          - Model: {payload.get('model')}")
                print(f"          - Features: {payload.get('features')}")
                print(f"          - Hasil Inferensi (Prediction): {payload.get('prediction')}")
            except:
                pass
            
            # Send verification response
            await websocket.send_text(json.dumps({
                "status": "synchronized",
                "message": "Data berhasil diamankan oleh Raksa Cloud",
                "received_prediction": payload.get('prediction') if 'payload' in locals() else None
            }))
    except WebSocketDisconnect:
        print("[BACKEND] Perangkat edge terputus.")
    except Exception as e:
        print(f"[BACKEND] Gangguan koneksi: {e}")

if __name__ == "__main__":
    # Run locally on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
