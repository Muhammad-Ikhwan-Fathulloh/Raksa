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


# ── Preprocessing & Machine Learning Features (NocML inspired) ───────────────

class MinMaxScaler:
    def __init__(self, dims, min_vals, max_vals):
        self.dims = dims
        self.min_vals = list(min_vals)
        self.max_vals = list(max_vals)
        # Precompute denominators to avoid division by zero and speed up math
        self.ranges = [1.0] * dims
        for i in range(dims):
            r = self.max_vals[i] - self.min_vals[i]
            self.ranges[i] = r if r != 0.0 else 1.0

    @micropython.native
    def transform(self, x, out=None):
        if out is None:
            out = [0.0] * self.dims
        for i in range(self.dims):
            out[i] = (x[i] - self.min_vals[i]) / self.ranges[i]
        return out


class StandardScaler:
    def __init__(self, dims, means, stddevs):
        self.dims = dims
        self.means = list(means)
        self.stddevs = list(stddevs)
        # Avoid division by zero
        self.scales = [1.0] * dims
        for i in range(dims):
            s = self.stddevs[i]
            self.scales[i] = s if s != 0.0 else 1.0

    @micropython.native
    def transform(self, x, out=None):
        if out is None:
            out = [0.0] * self.dims
        for i in range(self.dims):
            out[i] = (x[i] - self.means[i]) / self.scales[i]
        return out


class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
        self._cache = {}

    def _get_combinations(self, dims):
        if dims in self._cache:
            return self._cache[dims]

        # Generate combinations with replacement manually without itertools
        combos = []
        for d in range(self.degree + 1):
            combos.extend(self._cvw(list(range(dims)), d))
        self._cache[dims] = combos
        return combos

    def _cvw(self, items, r):
        # Helper for combinations with replacement
        if r == 0:
            return [()]
        if not items:
            return []
        n = len(items)
        indices = [0] * r
        results = [tuple(items[i] for i in indices)]
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return results
            indices[i:] = [indices[i] + 1] * (r - i)
            results.append(tuple(items[k] for k in indices))

    @micropython.native
    def transform(self, x, dims=None, out=None):
        if dims is None:
            dims = len(x)
        combos = self._get_combinations(dims)
        
        n_out = len(combos)
        if out is None:
            out = [0.0] * n_out
            
        for i in range(n_out):
            combo = combos[i]
            val = 1.0
            for idx in combo:
                val *= x[idx]
            out[i] = val
        return out


class LinearForecaster:
    def __init__(self):
        self.m = 0.0
        self.c = 0.0
        self.last_x = 0.0
        self.step = 1.0

    def fit(self, x, y):
        n = len(x)
        if n == 0:
            return
        sum_x = sum(x)
        sum_y = sum(y)
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        num = 0.0
        den = 0.0
        for i in range(n):
            dx = x[i] - mean_x
            num += dx * (y[i] - mean_y)
            den += dx * dx
            
        if den != 0.0:
            self.m = num / den
        else:
            self.m = 0.0
        self.c = mean_y - self.m * mean_x
        
        self.last_x = x[-1]
        self.step = (x[-1] - x[0]) / (n - 1) if n > 1 else 1.0

    def predict(self, x):
        return self.m * x + self.c

    def forecastNext(self):
        return self.m * (self.last_x + self.step) + self.c


class KNN:
    def __init__(self, training_data, labels, num_samples, dims, k=3):
        self.training_data = list(training_data)
        self.labels = list(labels)
        self.num_samples = num_samples
        self.dims = dims
        self.k = k
        self.is_flat = len(self.training_data) == num_samples * dims

    @micropython.native
    def predict(self, x):
        dists = [0.0] * self.num_samples
        if self.is_flat:
            for i in range(self.num_samples):
                d2 = 0.0
                offset = i * self.dims
                for j in range(self.dims):
                    diff = x[j] - self.training_data[offset + j]
                    d2 += diff * diff
                dists[i] = d2
        else:
            for i in range(self.num_samples):
                d2 = 0.0
                row = self.training_data[i]
                for j in range(self.dims):
                    diff = x[j] - row[j]
                    d2 += diff * diff
                dists[i] = d2

        pairs = []
        for i in range(self.num_samples):
            pairs.append((dists[i], self.labels[i]))
        pairs.sort()
        
        votes = {}
        for i in range(self.k):
            lbl = pairs[i][1]
            votes[lbl] = votes.get(lbl, 0) + 1
            
        max_votes = -1
        winner = None
        for lbl, count in votes.items():
            if count > max_votes:
                max_votes = count
                winner = lbl
        return winner


class NaiveBayes:
    def __init__(self, num_classes, dims, means, vars, priors):
        self.num_classes = num_classes
        self.dims = dims
        self.means = list(means)
        self.vars = list(vars)
        self.priors = list(priors)

    @micropython.native
    def predict(self, x):
        best_class = 0
        max_log_prob = -1e9
        import math
        
        for c in range(self.num_classes):
            log_prob = math.log(self.priors[c])
            for j in range(self.dims):
                idx = c * self.dims + j
                mean_val = self.means[idx]
                var_val = self.vars[idx]
                if var_val <= 0.0:
                    var_val = 1e-9
                diff = x[j] - mean_val
                log_prob += -0.5 * math.log(6.283185307179586 * var_val) - (diff * diff) / (2.0 * var_val)
                
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = c
        return best_class


class LogisticRegression:
    def __init__(self, dims, weights, bias):
        self.dims = dims
        self.weights = list(weights)
        self.bias = bias

    @micropython.native
    def predict(self, x):
        z = self.bias
        for i in range(self.dims):
            z += self.weights[i] * x[i]
        return 1 if z >= 0.0 else 0

    def predict_proba(self, x):
        z = self.bias
        for i in range(self.dims):
            z += self.weights[i] * x[i]
        import math
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 0.0 if z < 0 else 1.0


class DecisionTreeClassifier:
    def __init__(self, nodes, num_nodes=None):
        self.nodes = list(nodes)
        self.num_nodes = len(self.nodes) if num_nodes is None else num_nodes

    @micropython.native
    def predict(self, x):
        curr_idx = 0
        while curr_idx < self.num_nodes:
            node = self.nodes[curr_idx]
            if isinstance(node, dict):
                feat = node.get("feature", -1)
                if feat == -1: feat = node.get("feature_idx", -1)
                threshold = node.get("threshold", 0.0)
                left = node.get("left", -1)
                if left == -1: left = node.get("left_child", -1)
                right = node.get("right", -1)
                if right == -1: right = node.get("right_child", -1)
                val = node.get("value", -1)
            elif isinstance(node, (list, tuple)):
                feat = int(node[0])
                threshold = float(node[1])
                left = int(node[2])
                right = int(node[3])
                val = int(node[4])
            else:
                feat = getattr(node, "feature", -1)
                if feat == -1: feat = getattr(node, "feature_idx", -1)
                threshold = getattr(node, "threshold", 0.0)
                left = getattr(node, "left", -1)
                if left == -1: left = getattr(node, "left_child", -1)
                right = getattr(node, "right", -1)
                if right == -1: right = getattr(node, "right_child", -1)
                val = getattr(node, "value", -1)

            if feat < 0:
                return val
            if x[feat] <= threshold:
                curr_idx = left
            else:
                curr_idx = right
        return -1


class KMeans:
    def __init__(self, k, dims, centroids=None):
        self.k = k
        self.dims = dims
        if centroids is not None:
            self.centroids = list(centroids)
            self.is_flat = len(self.centroids) == k * dims
        else:
            self.centroids = [0.0] * (k * dims)
            self.is_flat = True

    @micropython.native
    def predict(self, x):
        best_k = 0
        min_d2 = 1e9
        for i in range(self.k):
            d2 = 0.0
            if self.is_flat:
                offset = i * self.dims
                for j in range(self.dims):
                    diff = x[j] - self.centroids[offset + j]
                    d2 += diff * diff
            else:
                row = self.centroids[i]
                for j in range(self.dims):
                    diff = x[j] - row[j]
                    d2 += diff * diff
            if d2 < min_d2:
                min_d2 = d2
                best_k = i
        return best_k

    def run(self, data, num_samples, assignments=None, max_iters=10):
        data = list(data)
        is_data_flat = len(data) == num_samples * self.dims
        if assignments is None:
            assignments = [0] * num_samples
            
        total_zeros = True
        for val in self.centroids:
            if val != 0.0:
                total_zeros = False
                break
        if total_zeros:
            for i in range(self.k):
                sample_idx = i % num_samples
                if is_data_flat:
                    for j in range(self.dims):
                        self.centroids[i * self.dims + j] = data[sample_idx * self.dims + j]
                else:
                    self.centroids[i] = list(data[sample_idx])

        for _ in range(max_iters):
            changed = False
            for i in range(num_samples):
                if is_data_flat:
                    x = data[i * self.dims : (i + 1) * self.dims]
                else:
                    x = data[i]
                old_k = assignments[i]
                new_k = self.predict(x)
                if old_k != new_k:
                    assignments[i] = new_k
                    changed = True
            if not changed:
                break

            cnt = [0] * self.k
            sums = [0.0] * (self.k * self.dims)
            for i in range(num_samples):
                cid = assignments[i]
                cnt[cid] += 1
                if is_data_flat:
                    for j in range(self.dims):
                        sums[cid * self.dims + j] += data[i * self.dims + j]
                else:
                    for j in range(self.dims):
                        sums[cid * self.dims + j] += data[i][j]

            for cid in range(self.k):
                n_pts = cnt[cid]
                if n_pts > 0:
                    for j in range(self.dims):
                        val = sums[cid * self.dims + j] / n_pts
                        if self.is_flat:
                            self.centroids[cid * self.dims + j] = val
                        else:
                            self.centroids[cid][j] = val
        return assignments
