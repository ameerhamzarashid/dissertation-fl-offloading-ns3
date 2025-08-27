# -*- coding: utf-8 -*-
import asyncio, json, os, socket, traceback, threading, time
from .tcp_protocol import send_json, recv_json

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 50051

def compute_update_bytes() -> int:
    num_params = int(os.environ.get("NUM_PARAMS", "1000000"))
    variant = os.environ.get("BRIDGE_VARIANT", "baseline").lower()
    if variant == "sfea":
        k = float(os.environ.get("K_PERCENT", "10"))
        kept = max(1, int(num_params * (k / 100.0)))
        return kept * 8  # ~4B value + 4B index
    else:
        return num_params * 4

def summarize_env() -> str:
    variant = os.environ.get("BRIDGE_VARIANT", "baseline").lower()
    num_params = os.environ.get("NUM_PARAMS", "1000000")
    k = os.environ.get("K_PERCENT", "-")
    try:
        upd = compute_update_bytes()
    except Exception:
        upd = -1
    return f"variant={variant} num_params={num_params} k_percent={k} update_bytes={upd}"

class BridgeServer:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        print(f"[bridge] client connected: {peer}")
        try:
            while True:
                try:
                    payload = await recv_json(reader)
                except asyncio.IncompleteReadError:
                    break
                msg = json.loads(payload)
                if msg.get("type") == "state":
                    action = 1 if msg["size"] <= 10*1024*1024 else 0
                    out = {"type":"action","ue":msg["ue"],"action":action,"update_bytes":compute_update_bytes()}
                    await send_json(writer, json.dumps(out))
        finally:
            print(f"[bridge] client disconnected: {peer}")
            writer.close()
            await writer.wait_closed()

    async def run(self):
        # Create a reusable listening socket to avoid bind failures caused by
        # TIME_WAIT or other transient socket states. Make the socket
        # non-blocking and hand it to asyncio.start_server via the `sock`
        # parameter so the event loop uses the prebound socket.
        try:
            listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_sock.bind((self.host, self.port))
            listen_sock.listen(100)
            listen_sock.setblocking(False)
        except Exception as e:
            print(f"[bridge] ERROR: failed to bind {self.host}:{self.port}: {e}")
            print("[bridge] HINT: check for other processes using the port: ss -ltnp | grep %d" % self.port)
            raise

        server = await asyncio.start_server(self.handle_client, sock=listen_sock)
        addrs = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
        print(f"[bridge] listening on {addrs}")
        print(f"[bridge] config: {summarize_env()}")
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    # Decide whether to attempt loading a Dueling-DQN model.
    variant = os.environ.get("BRIDGE_VARIANT", "baseline").lower()
    model_path = os.environ.get("MODEL_PATH")
    # We intentionally bind and start the listening socket immediately
    # (see BridgeServer.run) so the orchestrator can detect readiness even
    # when model loading is slow. Load the (potentially large) model in a
    # background thread and log progress; this eliminates the race where the
    # bridge process exists but never prints "listening" until a long model
    # load completes.

    def start_background_model_load(path: str):
        if not path or not os.path.exists(path):
            print("[bridge] No trained model loaded (MODEL_PATH not set or file missing) for baseline")
            return

        done = threading.Event()

        def reporter():
            # Periodic heartbeat while model is loading to make slow startups
            # visible in logs.
            while not done.is_set():
                print("[bridge] background model loader: still loading...")
                done.wait(5)

        def loader():
            start_t = time.time()
            print(f"[bridge] background model loader: starting load from {path}")
            rep = threading.Thread(target=reporter, daemon=True)
            rep.start()
            try:
                import torch
                from ..agents.dueling_dqn import DuelingQ, DQNConfig
                model = DuelingQ(obs_dim=3, n_actions=2)
                state = torch.load(path, map_location='cpu')
                model.load_state_dict(state)
                model.eval()
                elapsed = time.time() - start_t
                print(f"[bridge] background model loader: loaded model from {path} in {elapsed:.1f}s")
            except Exception as e:
                print(f"[bridge] WARNING (background load): Failed to load model from {path}: {e}")
                print("[bridge] HINT: This usually means the provided MODEL_PATH is not a DQN state_dict.")
                print("[bridge] HINT: If you intended to run SFEA, set BRIDGE_VARIANT=sfea and start the bridge again.")
                print("[bridge] HINT: If you exported MODEL_PATH intentionally for multi-seed, either unset MODEL_PATH before running the script (recommended) or run the script with EXPLICIT_MODEL_PATH=1 to force your path.")
            finally:
                done.set()

        t = threading.Thread(target=loader, daemon=True)
        t.start()

    if variant == "baseline":
        # Start model load in background if a model path is provided; do not
        # block startup on it.
        if model_path:
            start_background_model_load(model_path)
        else:
            print("[bridge] No MODEL_PATH provided; running with default/random policy for baseline")
    else:
        print(f"[bridge] BRIDGE_VARIANT={variant}; skipping DQN model load")

    try:
        asyncio.run(BridgeServer().run())
    except Exception:
        # Print a full traceback to the console so the multi-seed harness can
        # capture useful diagnostic output in per-run bridge logs.
        traceback.print_exc()
        print("[bridge] Exiting due to error during startup")
