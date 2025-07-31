#!/usr/bin/env python3
import socket, struct, logging, signal

HOST, PORT = '0.0.0.0', 12345
logging.basicConfig(level=logging.INFO,
    format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

running = True
def stop(signum, frame):
    global running; running = False

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

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
                    hdr = conn.recv(4)
                    if not hdr: break
                    length, = struct.unpack('!I', hdr)
                    if length != 40:
                        logging.error(f"Bad length {length}")
                        break
                    data = b''
                    while len(data) < length:
                        chunk = conn.recv(length - len(data))
                        if not chunk: break
                        data += chunk
                    if len(data)!=length:
                        logging.error("Incomplete payload")
                        break
                    vals = struct.unpack('!10I', data)
                    logging.info(f"State: {vals}")
                    action = sum(vals) % 256
                    conn.sendall(struct.pack('!I', action))
                    logging.info(f"Sent action {action}")
                except socket.timeout:
                    continue
                except Exception as e:
                    logging.error(f"Error: {e}")
                    break
        logging.info("Client disconnected")
    logging.info("Server shutting down")
