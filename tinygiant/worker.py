"""TinyGiant distributed worker — runs a subset of layers over TCP."""

import argparse
import json
import os

from ._constants import N_LAYERS
from .distributed import (
    MSG_FORWARD, MSG_RESET, MSG_PING, MSG_SHUTDOWN,
    MSG_RESULT, MSG_RESET_ACK, MSG_PONG,
    send_msg, recv_msg, unpack_forward,
    WorkerServer,
)
from .engine import NWSEngine


def cmd_worker():
    parser = argparse.ArgumentParser(
        prog="tinygiant-worker",
        description="TinyGiant distributed worker")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    with open(os.path.expanduser(args.config)) as f:
        cfg_data = json.load(f)

    model_path = os.path.expanduser(cfg_data["model"])
    cache_dir = os.path.expanduser(cfg_data["cache"])
    split_layer = cfg_data.get("split_layer", 24)
    listen_port = cfg_data.get("listen_port", 9100)
    pin_count = cfg_data.get("pin", 48)
    calibrate = cfg_data.get("calibrate", 10)

    # Load engine with worker's layer range — no embeddings, no output head
    engine = NWSEngine(
        model_path, cache_dir,
        layer_start=split_layer, layer_end=N_LAYERS,
        load_embeddings=False, load_output=False,
    )

    server = WorkerServer(listen_port)
    print(f"\n[worker] listening on 0.0.0.0:{listen_port}")
    print(f"[worker] layers {split_layer}-{N_LAYERS - 1}")

    calibrated = False

    def handle_msg(msg_type, payload, conn):
        nonlocal calibrated

        if msg_type == MSG_FORWARD:
            x, pos = unpack_forward(payload)
            for i in range(engine.layer_start, engine.layer_end):
                x = engine._layer_forward(x, i, pos) if (engine.is_q4 and engine.tg_lib) \
                    else engine._layer_forward_numpy(x, i, pos)
            engine.kv_len = pos + 1
            send_msg(conn, MSG_RESULT, x.tobytes())

            if not calibrated and pos >= calibrate:
                print(f"[worker] pinning top-{pin_count} experts/layer from usage")
                engine.expert_cache.pin_from_usage(pin_count, engine.layer_end)
                calibrated = True

            return True

        elif msg_type == MSG_RESET:
            engine.reset_kv()
            send_msg(conn, MSG_RESET_ACK, b"")
            return True

        elif msg_type == MSG_PING:
            send_msg(conn, MSG_PONG, b"")
            return True

        elif msg_type == MSG_SHUTDOWN:
            print("[worker] shutdown requested")
            return False

        else:
            print(f"[worker] unknown msg_type={msg_type}")
            return True

    try:
        server.serve_forever(handle_msg)
    except KeyboardInterrupt:
        print("\n[worker] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    cmd_worker()
