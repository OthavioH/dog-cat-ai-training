import socket

def connect_client_to_server(host = 'localhost', port = 27010):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    client.send(str("Hello from client").encode())
    while True:
        data = client.recv(1024).decode()
        if data:
            print(f"Received {data} from coordinator")