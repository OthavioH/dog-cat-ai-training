import itertools
import socket
import json
import time
from flask import Flask, request, jsonify

from ai_parameters import AIParameters

clients_connected = {}
client_status = {client: "free" for client in clients_connected}
param_queue: list[AIParameters] = []
task_start_times = {}

# create an API with POST /train
app = Flask(__name__)

def handle_client(conn):
    data = conn.recv(1024).decode()
    if data:
        client_address = conn.getpeername()
        print(f"Received {data} from {client_address}")

        json_data = json.loads(data)
        # verify if the message is "action:connect"
        if json_data.get("action") == "connect":
            if client_address not in clients_connected:
                clients_connected[client_address] = conn
                global client_status
                client_status = {client: "free" for client in clients_connected}
                print("Client connected")
            else:
                print("Client already connected")
        elif json_data.get("action") == "finishedProcessing":
            client_status[client_address] = "free"
            print(f"Set status to free for {client_address}")
            on_client_finish(client_address, json_data)
        elif json_data.get("action") == "setStatusToBusy":
            client_status[client_address] = "busy"
            print(f"Set status to busy for {client_address}")

def openSocket(host, socketPort):
    print("Starting coordinator...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, socketPort))    # Start the coordinator

    server.listen(5)

    print(f"Coordinator listening on {host}")

    while True:
        conn, _ = server.accept()
        handle_client(conn)

def start_coordinator(host='0.0.0.0'):
    openSocket(host, 27010)
    
    app.run(host='0.0.0.0', port=3000)

def get_ai_parameters_list():
    model_names = ['Alexnet', 'VGG11', 'MobilenetV3Large', 'MobilenetV3Small', 'Resnet18', 'Resnet101', 'VGG19'] 
    epochs = [5, 10]
    learning_rates = [0.001, 0.0001, 0.00001]
    weight_decays = [0, 0.0001]

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(model_names, epochs, learning_rates, weight_decays))
    # print(param_combinations)
    # Create a list of AIParameters instances
    # Create a list of AIParameters instances
    ai_parameters_list = []
    for params in param_combinations:
        json_data = {
            'model_name': params[0],
            'epoch': params[1],
            'learning_rate': params[2],
            'weight_decays': params[3]
        }
        ai_parameters_list.append(AIParameters(json_data))
    
    return ai_parameters_list

def distribute_tasks():
    while param_queue:
        for client in clients_connected:
            if client_status[client] == "free" and param_queue:
                params = param_queue.pop(0)
                send_task_to_client(client, params)
                client_status[client] = "busy"
        time.sleep(1)  


def send_task_to_client(client, params):
    # Simulate sending task to client
    print(f"Sending {params} to {client}")
    task_start_times[client] = time.time()
    client_conn = clients_connected[client]
    client_conn.send(json.dumps({"action": "process","params": params}).encode())
    

def on_client_finish(client,data):
    # Process the result received from the client
    # Save the result to a file or database

    end_time = time.time()
    start_time = task_start_times.pop(client, None)
    duration = end_time - start_time if start_time else None
    data['duration'] = duration
    with open('experiment_results.json', 'a') as f:
        json.dump(data, f)
        f.write('\n')

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()

    # verify if data has the message "action:train"
    if data.get("action") != "train":
        return jsonify({"status": "invalid data"}), 400
    # if data.get("isMultiprocessing") == True:
    #     print("Multiprocessing")
    # print(f"Training data received: {data}")
    # return jsonify({"status": "training started"}), 200
    global param_queue
    param_queue = get_ai_parameters_list()

    distribute_tasks()

if __name__ == '__main__':
    start_coordinator()