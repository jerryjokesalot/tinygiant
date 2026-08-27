"""Integration test — coordinator sends hidden state to worker, gets result back."""
import threading

import numpy as np
import pytest

from tinygiant._constants import EMBED_DIM
from tinygiant.distributed import (
    WorkerConnection, WorkerServer,
    MSG_FORWARD, MSG_RESULT, MSG_RESET, MSG_RESET_ACK,
    send_msg, recv_msg, unpack_forward,
)


def test_distributed_forward_mock_worker():
    """Coordinator forwards hidden state; mock worker adds 1.0 and returns."""
    server = WorkerServer(0)
    port = server._sock.getsockname()[1]

    def worker_loop():
        conn = server.accept()
        while True:
            try:
                msg_type, payload = recv_msg(conn)
            except ConnectionError:
                break
            if msg_type == MSG_FORWARD:
                x, pos = unpack_forward(payload)
                send_msg(conn, MSG_RESULT, (x + 1.0).tobytes())
            elif msg_type == MSG_RESET:
                send_msg(conn, MSG_RESET_ACK, b"")
            else:
                break
        conn.close()

    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

    wc = WorkerConnection()
    wc.connect("127.0.0.1", port)

    # Simulate 3 tokens
    for pos in range(3):
        x = np.full(EMBED_DIM, float(pos), dtype=np.float32)
        result = wc.forward(x, pos)
        np.testing.assert_allclose(result, x + 1.0)

    wc.reset()
    wc.close()
    server.close()
