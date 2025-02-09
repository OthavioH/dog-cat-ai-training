import socket
import multiprocessing

def handle_client(conn):
    """Recebe operações do cliente, processa e retorna o resultado"""
    data = conn.recv(1024).decode()
    if data:
        # get client address
        client_address = conn.getpeername()
        print(f"got {data} from {client_address}")
    #     operations = data.split(";")
    #     result = process_local(operations)
        conn.send(str('connected succesfully').encode())
    conn.close()

def start_server(host="0.0.0.0", port=27010):
    """Inicia o servidor para receber cálculos remotos"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Servidor rodando em {host}:{port}")

    while True:
        conn, _ = server.accept()
        handle_client(conn)
        # multiprocessing.Process(target=handle_client, args=(conn,)).start()
        
        
if __name__ == '__main__':
    start_server()
