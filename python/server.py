#!/usr/bin/env python3
import socket
import struct
import logging
import signal

# ——— Configuration ———
HOST, PORT = '127.0.0.1', 12345
LOG_FORMAT = '[%(asctime)s] %(message)s'
DATE_FORMAT = '%H:%M:%S'

# ——— Setup Logging ———
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# Graceful shutdown flag
running = True
def stop(signum, frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

# ——— Main Server Loop ———
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    logging.info(f"Listening on {HOST}:{PORT}")
    srv.settimeout(1.0)
    while running:
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        logging.info(f"Client connected: {addr}")
        with conn:
            conn.settimeout(1.0)
            while running:
                try:
                    data = conn.recv(4)
                    if not data:
                        break
                    (n,) = struct.unpack('!I', data)
                    logging.info(f"Recv ⟶ {n}")
                    n += 1
                    conn.sendall(struct.pack('!I', n))
                    logging.info(f"Send ⟵ {n}")
                except socket.timeout:
                    continue
        logging.info("Client disconnected")
    logging.info("Server shutting down")
