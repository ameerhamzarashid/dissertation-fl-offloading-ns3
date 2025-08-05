#!/usr/bin/env python3
"""
Baseline Test Client for FL Server
Simple client to test the federated learning server
"""
import socket
import struct
import time

def test_baseline_client():
    """Test client for baseline FL server"""
    HOST = '127.0.0.1'
    PORT = 12345
    
    print("Connecting to baseline FL server...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))
            print(f"Connected to {HOST}:{PORT}")
            
            # Send test states representing device utilization
            test_states = [25.0, 45.0, 75.0, 35.0, 85.0]  # Different utilization levels
            
            for i, state in enumerate(test_states):
                print(f"\n--- Test {i+1} ---")
                
                # Send state as 4-byte network byte order float
                network_state = struct.pack('!f', state)
                network_state_as_int = struct.unpack('!I', network_state)[0]
                sock.sendall(struct.pack('!I', network_state_as_int))
                
                print(f"Sent state: {state} (utilization %)")
                
                # Receive action
                action_data = sock.recv(4)
                if len(action_data) == 4:
                    action = struct.unpack('!I', action_data)[0]
                    action_names = {0: "Local", 1: "Edge", 2: "Cloud"}
                    print(f"Received action: {action} ({action_names.get(action, 'Unknown')})")
                else:
                    print("Failed to receive action")
                    break
                
                time.sleep(1)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    test_baseline_client()
