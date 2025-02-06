import socket

def handle_client(conn):
    data = conn.recv(1024).decode()
    if data:
        client_address = conn.getpeername()
        print(f"Received {data} from {client_address}")

        conn.send(str("Hello from coordinator").encode())
    conn.close()


def start_coordinator(host='0.0.0.0', port = 27010):
    print("Starting coordinator...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))    # Start the coordinator

    server.listen(5)

    print(f"Coordinator listening on {host}")

    while True:
        conn, _ = server.accept()
        handle_client(conn)

if __name__ == '__main__':
    start_coordinator()