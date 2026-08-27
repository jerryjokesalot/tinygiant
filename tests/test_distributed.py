import socket
import struct
import threading
import numpy as np
import pytest

from tinygiant.distributed import (
    MSG_FORWARD, MSG_RESULT, MSG_RESET, MSG_RESET_ACK,
    MSG_PING, MSG_PONG, MSG_SHUTDOWN,
    send_msg, recv_msg, pack_forward, unpack_forward,
    WorkerConnection, WorkerServer,
)


def test_msg_constants():
    assert MSG_FORWARD == 1
    assert MSG_RESULT == 2
    assert MSG_RESET == 3
    assert MSG_RESET_ACK == 4
    assert MSG_PING == 5
    assert MSG_PONG == 6
    assert MSG_SHUTDOWN == 7


def test_pack_unpack_forward():
    x = np.random.randn(2048).astype(np.float32)
    pos = 42
    payload = pack_forward(x, pos)
    assert len(payload) == 4 + 2048 * 4  # int32 + float32[2048]
    x_out, pos_out = unpack_forward(payload)
    assert pos_out == 42
    np.testing.assert_array_equal(x_out, x)


def test_pack_unpack_forward_pos_zero():
    x = np.zeros(2048, dtype=np.float32)
    payload = pack_forward(x, 0)
    x_out, pos_out = unpack_forward(payload)
    assert pos_out == 0
    np.testing.assert_array_equal(x_out, x)


def test_send_recv_msg_over_socket():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    received = {}

    def server_thread():
        conn, _ = server_sock.accept()
        msg_type, payload = recv_msg(conn)
        received["type"] = msg_type
        received["payload"] = payload
        send_msg(conn, MSG_PONG, b"")
        conn.close()
        server_sock.close()

    t = threading.Thread(target=server_thread)
    t.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    send_msg(client, MSG_PING, b"")
    msg_type, payload = recv_msg(client)
    client.close()
    t.join()

    assert received["type"] == MSG_PING
    assert received["payload"] == b""
    assert msg_type == MSG_PONG
    assert payload == b""


def test_send_recv_forward_payload():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    x_in = np.random.randn(2048).astype(np.float32)

    def server_thread():
        conn, _ = server_sock.accept()
        msg_type, payload = recv_msg(conn)
        x, pos = unpack_forward(payload)
        # Echo back with doubled values
        send_msg(conn, MSG_RESULT, (x * 2).tobytes())
        conn.close()
        server_sock.close()

    t = threading.Thread(target=server_thread)
    t.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    send_msg(client, MSG_FORWARD, pack_forward(x_in, 7))
    msg_type, payload = recv_msg(client)
    x_out = np.frombuffer(payload, dtype=np.float32)
    client.close()
    t.join()

    assert msg_type == MSG_RESULT
    np.testing.assert_allclose(x_out, x_in * 2)


def test_worker_connection_forward_round_trip():
    """Full coordinator→worker forward round-trip over localhost."""
    server = WorkerServer(0)  # OS-assigned port
    port = server._sock.getsockname()[1]

    x_in = np.random.randn(2048).astype(np.float32)

    def worker_thread():
        conn = server.accept()
        msg_type, payload = recv_msg(conn)
        x, pos = unpack_forward(payload)
        send_msg(conn, MSG_RESULT, (x + 1.0).tobytes())
        # Wait for reset
        msg_type, _ = recv_msg(conn)
        send_msg(conn, MSG_RESET_ACK, b"")
        conn.close()

    t = threading.Thread(target=worker_thread)
    t.start()

    wc = WorkerConnection()
    wc.connect("127.0.0.1", port)
    x_out = wc.forward(x_in, 5)
    wc.reset()
    wc.close()
    server.close()
    t.join()

    np.testing.assert_allclose(x_out, x_in + 1.0)


def test_worker_connection_ping():
    server = WorkerServer(0)
    port = server._sock.getsockname()[1]

    def worker_thread():
        conn = server.accept()
        msg_type, _ = recv_msg(conn)
        assert msg_type == MSG_PING
        send_msg(conn, MSG_PONG, b"")
        conn.close()

    t = threading.Thread(target=worker_thread)
    t.start()

    wc = WorkerConnection()
    wc.connect("127.0.0.1", port)
    assert wc.ping() is True
    wc.close()
    server.close()
    t.join()
