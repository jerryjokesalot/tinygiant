"""Distributed inference protocol — coordinator/worker over TCP.

Wire format: [uint32 msg_type][uint32 payload_len][payload bytes]
Hidden state: int32 pos + float32[2048] = 8196 bytes per FORWARD.
"""

import socket
import struct
import threading
import time

import numpy as np

from ._constants import EMBED_DIM

# Message types
MSG_FORWARD = 1
MSG_RESULT = 2
MSG_RESET = 3
MSG_RESET_ACK = 4
MSG_PING = 5
MSG_PONG = 6
MSG_SHUTDOWN = 7

_HEADER = struct.Struct("!II")  # msg_type, payload_len — network byte order
_POS = struct.Struct("!i")      # int32 pos


def send_msg(sock, msg_type, payload):
    header = _HEADER.pack(msg_type, len(payload))
    sock.sendall(header + payload)


def recv_msg(sock):
    header = _recv_exact(sock, _HEADER.size)
    msg_type, payload_len = _HEADER.unpack(header)
    payload = _recv_exact(sock, payload_len) if payload_len > 0 else b""
    return msg_type, payload


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        buf.extend(chunk)
    return bytes(buf)


def pack_forward(x, pos):
    return _POS.pack(pos) + x.tobytes()


def unpack_forward(payload):
    pos = _POS.unpack_from(payload, 0)[0]
    x = np.frombuffer(payload, dtype=np.float32, offset=_POS.size).copy()
    return x, pos


class WorkerConnection:
    """Coordinator-side handle to a remote layer worker."""

    def __init__(self):
        self._sock = None
        self._lock = threading.Lock()

    def connect(self, host, port, timeout=10):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((host, port))
        self._sock = sock

    @property
    def connected(self):
        return self._sock is not None

    def forward(self, x, pos):
        with self._lock:
            send_msg(self._sock, MSG_FORWARD, pack_forward(x, pos))
            msg_type, payload = recv_msg(self._sock)
            if msg_type != MSG_RESULT:
                raise RuntimeError(f"Expected RESULT, got msg_type={msg_type}")
            return np.frombuffer(payload, dtype=np.float32).copy()

    def reset(self):
        with self._lock:
            send_msg(self._sock, MSG_RESET, b"")
            msg_type, _ = recv_msg(self._sock)
            if msg_type != MSG_RESET_ACK:
                raise RuntimeError(f"Expected RESET_ACK, got msg_type={msg_type}")

    def ping(self):
        try:
            with self._lock:
                send_msg(self._sock, MSG_PING, b"")
                self._sock.settimeout(10)
                msg_type, _ = recv_msg(self._sock)
                return msg_type == MSG_PONG
        except (OSError, ConnectionError):
            return False

    def shutdown(self):
        if self._sock:
            try:
                send_msg(self._sock, MSG_SHUTDOWN, b"")
            except OSError:
                pass

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class WorkerServer:
    """Worker-side TCP server that accepts one coordinator connection."""

    def __init__(self, port, host="0.0.0.0"):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(1)
        self.port = port
        self.host = host

    def accept(self):
        conn, addr = self._sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[worker] coordinator connected from {addr[0]}:{addr[1]}")
        return conn

    def serve_forever(self, handler):
        """Loop: accept connection, dispatch messages to handler until disconnect.

        handler is called as handler(msg_type, payload, conn) and must return
        True to continue or False to shut down.
        """
        while True:
            print(f"[worker] listening on {self.host}:{self.port}")
            conn = self.accept()
            try:
                while True:
                    msg_type, payload = recv_msg(conn)
                    if not handler(msg_type, payload, conn):
                        return
            except ConnectionError:
                print("[worker] coordinator disconnected, waiting for reconnect")
            finally:
                conn.close()

    def close(self):
        self._sock.close()
