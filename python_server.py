# python_server.py
import socket
import struct

HOST = '127.0.0.1'
PORT = 12345
BUFFER_SIZE = 1024

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"--- Python TCP Server is listening on {HOST}:{PORT} ---")

conn, addr = server_socket.accept()
print(f"--- Connection established from ns-3 at {addr} ---")

try:
    while True:
        data = conn.recv(BUFFER_SIZE)
        if not data:
            print("--- ns-3 client closed the connection. ---")
            break
        
        received_int = struct.unpack('!i', data)[0]
        print(f"[Python] Received integer from ns-3: {received_int}")

        processed_int = received_int + 1
        print(f"[Python] Processed value: {processed_int}")

        response_data = struct.pack('!i', processed_int)
        conn.sendall(response_data)
        print(f"[Python] Sent response back to ns-3: {processed_int}")
        print("-" * 30)
except Exception as e:
    print(f"[ERROR] An error occurred: {e}")
finally:
    print("--- Closing connection and server socket. ---")
    conn.close()
    server_socket.close()