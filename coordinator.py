import itertools
import socket
import json
import threading
import time
from flask import Flask, request, jsonify

from ai_parameters import AIParameters
from cnn import CNN
from multi_thread_trainer import MultiThreadTrainer

class Coordinator:
    clients_connected: dict[str,any] = {}
    param_queue: list[AIParameters] = []
    task_start_times = {}

    app = Flask(__name__)
    
    cnn: CNN = None
    
    def __init__(self, cnn:CNN, param_list: list[AIParameters]):
        self.clients_connected = {
            "myself": "localhost",
        }
        self.param_queue = param_list
        self.cnn = cnn

    def handle_client(self,conn):
        data = conn.recv(1024).decode()
        if data:
            client_address = conn.getpeername()
            print(f"Received {data} from {client_address}")

            json_data = json.loads(data)
            if json_data.get("action") == "connect":
                if client_address not in self.clients_connected:
                    self.clients_connected[client_address] = conn
                    print("Client connected")
                else:
                    print("Client already connected")
            elif json_data.get("action") == "finishedProcessing":
                print(f"Set status to free for {client_address}")
                self.on_client_finish_one_parameter(client_address, json_data)

    def openSocket(self,host, socketPort):
        print("Starting coordinator...")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, socketPort))    # Start the coordinator

        server.listen(5)

        print(f"Coordinator listening on {host}:{socketPort}")

        while True:
            conn, _ = server.accept()
            self.handle_client(conn)

    def start_coordinator(self,host='0.0.0.0'):
        @self.app.route('/train', methods=['POST'])
        def train():
            json_data = request.get_data(as_text=True)
            print(f"Received data: {json_data}")
            data = json.loads(json_data)

            if data.get("action") != "train":
                return jsonify({"status": "invalid data"}), 400

            self.distribute_tasks()

        def run_flask():
            self.app.run(host=host, port=3000)
            print("Coordinator API running on port 3000")
        
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.start()
        
        self.openSocket(host, 27010)
        

    def distribute_tasks(self):
        clients_param_queues = { client: [] for client in self.clients_connected }
        while self.param_queue:
            for client in self.clients_connected:
                params = self.param_queue.pop(0)
                clients_param_queues[client].append(params)
        for client, params in clients_param_queues.items():
            if client == "myself":
                pass
            else:
                self.send_task_to_client(client, params)
        self.process_on_the_same_machine(clients_param_queues.get("myself"))

    def process_on_the_same_machine(self,params:list[AIParameters]):
        multi_thread_trainer = MultiThreadTrainer(params)
        multi_thread_trainer.process(cnn=self.cnn,must_save_results=True)

    def send_task_to_client(self,client, params:list[AIParameters]):
        try:
            print(f"Sending {params} to {client}")
            self.task_start_times[client] = time.time()
            client_conn = self.clients_connected[client]
            client_conn.send(json.dumps({"action": "process","params": [
                param.to_json() for param in params
            ]}).encode())
        except Exception as e:
            print(f"Error sending task to client {client}: {e}")
        

    def on_client_finish_one_parameter(self,client,data):
            
        end_time = time.time()
        start_time = self.task_start_times.pop(client, None)
        duration = end_time - start_time if start_time else None
        data['duration'] = duration
        with open('distributed_results.json', 'a') as f:
            json.dump(data, f)
            f.write('\n')