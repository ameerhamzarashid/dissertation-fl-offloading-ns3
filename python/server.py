import socket, struct, pickle
import threading, logging
import torch, numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger("FLServer")

HOST, PORT = '0.0.0.0', 12345

def handle_client(conn, addr):
    logger.info(f"Client connected: {addr}")
    try:
        while True:
            # read 4-byte length
            data = conn.recv(4)
            if not data: break
            (length,) = struct.unpack('!I', data)
            payload = conn.recv(length)
            state = pickle.loads(payload)
            # dummy policy: action = sum(state) mod 10
            action = int(np.sum(state) % 10)
            logger.info(f"Recv ⟶ {state[0]}, Send ⟵ {action}")
            conn.sendall(struct.pack('!I', action))
    finally:
        conn.close()
        logger.info("Client disconnected")

def main():
    sock = socket.socket()
    sock.bind((HOST, PORT))
    sock.listen()
    logger.info(f"Listening on {HOST}:{PORT}")
    while True:
        conn, addr = sock.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == '__main__':
    main()
# End of server.py