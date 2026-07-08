# raksa.py
# MicroPython Client Library for In-situ Analytics and WebSocket communication.
# Part of the 'Noc Lab' TinyML ecosystem for the 'Tiny Chip, Big Brain' workshop.
# Developed with Nusantara values: Raksa = Protection, Guard, Reliability.

import gc

try:
    import ustruct as struct
except ImportError:
    import struct

try:
    import uasyncio as asyncio
except ImportError:
    import asyncio

try:
    import urandom
except ImportError:
    import random as urandom

# Dual compatibility for CPython and MicroPython platforms
try:
    import micropython
except ImportError:
    class micropython:
        @staticmethod
        def native(f):
            return f
        @staticmethod
        def viper(f):
            return f

# Standard Base64 characters helper for pure Python/MicroPython compilation
_B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

@micropython.native
def _b64encode(b: bytes) -> str:
    """Encode bytes to base64 string without external dependencies."""
    res = ""
    n = len(b)
    for i in range(0, n, 3):
        chunk = b[i:i+3]
        pad = 3 - len(chunk)
        if pad == 1:
            chunk = chunk + b'\x00'
        elif pad == 2:
            chunk = chunk + b'\x00\x00'
        
        val = (chunk[0] << 16) | (chunk[1] << 8) | chunk[2]
        res += _B64_CHARS[(val >> 18) & 63]
        res += _B64_CHARS[(val >> 12) & 63]
        res += _B64_CHARS[(val >> 6) & 63] if pad < 2 else "="
        res += _B64_CHARS[val & 63] if pad < 1 else "="
    return res


class RaksaClient:
    """
    RaksaClient: Handles persistent, asynchronous WebSocket connections on MicroPython.
    Manages client handshake, data serialization, memory optimization, and fast In-situ Analytics.
    """
    def __init__(self, uri: str, reconnect_delay: int = 5):
        self.uri = uri
        self.reconnect_delay = reconnect_delay
        self.host = ""
        self.port = 80
        self.path = "/"
        self.ssl = False
        self.reader = None
        self.writer = None
        self._connected = False
        self._lock = asyncio.Lock()
        
        self._parse_uri(uri)

    def _parse_uri(self, uri: str):
        """Parse incoming ws:// or wss:// link into connection details."""
        if not (uri.startswith("ws://") or uri.startswith("wss://")):
            raise ValueError("URL must start with ws:// or wss://")
            
        ssl = uri.startswith("wss://")
        self.ssl = ssl
        self.port = 443 if ssl else 80
        
        url_path = uri.split("://", 1)[1]
        if "/" in url_path:
            host_port, path = url_path.split("/", 1)
            self.path = "/" + path
        else:
            host_port = url_path
            self.path = "/"
            
        if ":" in host_port:
            host, port = host_port.split(":", 1)
            self.host = host
            self.port = int(port)
        else:
            self.host = host_port

    async def connect(self) -> bool:
        """Establish WebSocket connection with the cloud server."""
        async with self._lock:
            if self._connected:
                return True
                
            print(f"[RAKSA] Menghubungkan ke backend di {self.host}:{self.port} ...")
            try:
                # Async TCP socket connection
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                
                # Perform handshakes manually to keep codebase extremely tiny
                # Generate random 16-byte base64 key
                raw_key = bytes([urandom.getrandbits(8) for _ in range(16)])
                ws_key = _b64encode(raw_key)
                
                handshake = (
                    f"GET {self.path} HTTP/1.1\r\n"
                    f"Host: {self.host}:{self.port}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {ws_key}\r\n"
                    "Sec-WebSocket-Version: 13\r\n\r\n"
                )
                
                self.writer.write(handshake.encode("utf-8"))
                await self.writer.drain()
                
                # Verify switching protocols status line
                line = await self.reader.readline()
                if not line:
                    raise Exception("Koneksi ditutup secara tidak terduga oleh server.")
                    
                status_line = line.decode("utf-8")
                if "101" not in status_line:
                    raise Exception(f"Gagal melakukan upgrade websocket: {status_line.strip()}")
                
                # Consume response headers until double CR/LF
                while True:
                    line = await self.reader.readline()
                    if line == b"\r\n" or not line:
                        break
                        
                self._connected = True
                print("[RAKSA] Koneksi persisten berhasil dijalankan. Aliran data aman dan stabil.")
                gc.collect()
                return True
            except Exception as e:
                print(f"[RAKSA] Gagal menghubungkan: {e}")
                self._connected = False
                self.writer = None
                self.reader = None
                return False

    async def sync(self, payload) -> bool:
        """
        Sends telemetry/sensor payloads asynchronously via WebSocket.
        Invokes memory clean-up automatically in the cycles to avoid RAM fragmentation.
        """
        if not self._connected:
            success = await self.connect()
            if not success:
                return False

        # Parse dict to JSON string if applicable
        if isinstance(payload, dict):
            import json
            try:
                payload = json.dumps(payload)
            except Exception as e:
                print(f"[RAKSA] Kesalahan serialisasi JSON: {e}")
                return False

        if not isinstance(payload, str):
            payload = str(payload)

        async with self._lock:
            try:
                # WebSocket Frame specification
                data_bytes = payload.encode("utf-8")
                length = len(data_bytes)
                
                header = bytearray([0x81]) # Text frame FIN
                
                # Client to Server must be masked
                if length <= 125:
                    header.append(0x80 | length)
                elif length <= 65535:
                    header.append(0x80 | 126)
                    header.extend(struct.pack(">H", length))
                else:
                    header.append(0x80 | 127)
                    header.extend(struct.pack(">Q", length))
                
                # Generate random 4-byte mask
                mask_key = bytes([urandom.getrandbits(8) for _ in range(4)])
                header.extend(mask_key)
                
                masked_payload = self._mask_payload(data_bytes, mask_key)
                
                # Send frame header and payload
                self.writer.write(header)
                self.writer.write(masked_payload)
                await self.writer.drain()
                
                # Memory consolidation
                masked_payload = None
                header = None
                gc.collect() # Force execution of gc to maintain clean heap
                return True
                
            except Exception as e:
                print(f"[RAKSA] Pengiriman gagal: {e}. Mencoba memulihkan...")
                self._connected = False
                self.writer = None
                self.reader = None
                gc.collect()
                return False

    @micropython.native
    def _mask_payload(self, data: bytes, mask: bytes) -> bytes:
        """XOR mask payload for client packets. Native acceleration applied."""
        n = len(data)
        m = bytearray(n)
        for i in range(n):
            m[i] = data[i] ^ mask[i % 4]
        return bytes(m)

    @micropython.native
    def infer(self, model, data) -> float:
        """
        Process In-situ Analytics of data relative to a local ML weight model.
        Uses MicroPython `@micropython.native` decorator for maximum computation speed.
        
        Args:
            model: A model configuration, standard callable, or a tuple of (weights, biases) matrix.
            data: Data feature vector input (list).
            
        Returns:
            Computed inference result.
        """
        # Feature Matrix Multiply Inference Optimization
        # weights: list of float lists (outputs x inputs), biases: list of float values (outputs)
        if isinstance(model, tuple) and len(model) == 2:
            weights = model[0]
            biases = model[1]
            if isinstance(weights, list) and isinstance(biases, list) and isinstance(data, list):
                n_out = len(weights)
                n_in = len(data)
                
                if n_out > 0 and len(weights[0]) == n_in:
                    results = [0.0] * n_out
                    for i in range(n_out):
                        row = weights[i]
                        val = biases[i]
                        for j in range(n_in):
                            val += row[j] * data[j]
                        # Apply ReLU Activation
                        if val < 0.0:
                            results[i] = 0.0
                        else:
                            results[i] = val
                    return results
        
        # Standard model execution fallback checks
        if hasattr(model, 'predict') and callable(model.predict):
            return model.predict(data)
            
        if callable(model):
            return model(data)
            
        # Simulative linear TinyML model inference calculation fallback
        val_sum = 0.0
        n_features = len(data)
        for i in range(n_features):
            val_sum += data[i] * 0.45
        val_sum += 0.05
        return val_sum

    async def recv(self) -> str:
        """
        Receives a single WebSocket frame asynchronously and returns its text payload.
        Handles standard WebSocket framing including extended 16-bit and 64-bit lengths.
        """
        if not self._connected:
            raise Exception("[RAKSA] Client tidak terhubung.")
            
        try:
            # Read first 2 bytes (header)
            header = await self.reader.readexactly(2)
            opcode = header[0] & 0x0f
            has_mask = header[1] & 0x80
            length = header[1] & 0x7f
            
            # Close frame or ping/pong handlers
            if opcode == 0x08: # CLOSE
                await self.close()
                return ""
            elif opcode == 0x09: # PING
                # Send PONG frame
                await self._send_pong()
                return await self.recv()
            elif opcode == 0x0a: # PONG
                return await self.recv()
                
            # Parse extended lengths
            if length == 126:
                ext_len_bytes = await self.reader.readexactly(2)
                length = struct.unpack(">H", ext_len_bytes)[0]
            elif length == 127:
                ext_len_bytes = await self.reader.readexactly(8)
                length = struct.unpack(">Q", ext_len_bytes)[0]
                
            # Read mask key if server masked it (though servers shouldn't mask)
            if has_mask:
                mask_key = await self.reader.readexactly(4)
                
            # Read payload body
            body = await self.reader.readexactly(length)
            
            if has_mask:
                body = self._mask_payload(body, mask_key)
                
            # Decode and execute GC
            res = body.decode("utf-8")
            body = None
            gc.collect()
            return res
            
        except Exception as e:
            print(f"[RAKSA] Koneksi terputus saat membaca data: {e}")
            self._connected = False
            self.writer = None
            self.reader = None
            gc.collect()
            raise e

    async def _send_pong(self):
        """Send a WebSocket Pong frame in response to Ping."""
        async with self._lock:
            if self.writer:
                try:
                    self.writer.write(b'\x8a\x00') # FIN + PONG, length 0
                    await self.writer.drain()
                except:
                    pass

    async def close(self):
        """Close connection cleanly and run gc.collect."""
        async with self._lock:
            if self.writer:
                try:
                    self.writer.close()
                    await self.writer.wait_closed()
                except:
                    pass
            self.writer = None
            self.reader = None
            self._connected = False
            gc.collect()
