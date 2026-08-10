"""Minimal Chrome DevTools Protocol client over raw sockets (stdlib only).

Duoc dung de dieu khien trang /chat that: go cau hoi, cho tra loi, chup anh.
Khong fabricate DOM; moi thu chay tren Flask that va provider that.
"""
import base64
import json
import os
import socket
import struct
import time
import urllib.request


def http_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


class WS:
    """Client WebSocket rut gon: chi text frame, khong fragment, khong permessage-deflate."""

    def __init__(self, url):
        assert url.startswith("ws://")
        rest = url[len("ws://"):]
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port or 80)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        assert b"101" in buf.split(b"\r\n")[0], buf[:200]
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self.msg_id = 0

    def _recv_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("socket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, payload: dict):
        self.msg_id += 1
        payload["id"] = self.msg_id
        data = json.dumps(payload).encode("utf-8")
        header = bytearray([0x81])
        mask = os.urandom(4)
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(bytes(header) + masked)
        return self.msg_id

    def recv(self):
        b0, b1 = self._recv_exact(2)
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._recv_exact(8))[0]
        data = self._recv_exact(length)
        return json.loads(data.decode("utf-8"))

    def call(self, method, params=None, timeout=120):
        mid = self.send({"method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv()
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(method)

    def evaluate(self, expr, timeout=120):
        res = self.call(
            "Runtime.evaluate",
            {"expression": expr, "awaitPromise": True, "returnByValue": True},
            timeout=timeout,
        )
        if res.get("exceptionDetails"):
            raise RuntimeError(res["exceptionDetails"])
        return res["result"].get("value")