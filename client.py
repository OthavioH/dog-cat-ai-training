import json
import socket
import multiprocessing
from ai_parameters import AIParameters


fila_processamento_parametros: list[AIParameters] = []
computer_id = "NOTE_UEEK"

def connect_to_socket_server():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('10.151.35.182', 27010))
    json = {
        "action": "connect",
        "computer_id": computer_id
    }
    json_encoded = json.dumps(json)
    client.send(json_encoded.encode('utf-8'))
    while True:
        # get data sent from server
        data = client.recv(1024).decode()
        if data:
            print(data)
            json_data = json.loads(data)
            if(json_data.get('action') == 'process'):
                client.close()
                for json_ai_parameters in json_data.get('params'):
                    ai_params = AIParameters(json_ai_parameters)
                    fila_processamento_parametros.append(ai_params)
                with multiprocessing.Pool() as pool:
                    pool.map(process_ai_parameters, fila_processamento_parametros)
                    
def process_ai_parameters(ai_params:AIParameters):
    result = ai_params.train_ai(computer_id)
    send_result_to_server(result)
    fila_processamento_parametros.remove(ai_params) # Remove parametro da lista
                
            
def send_result_to_server(result:str):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('10.151.35.182', 27010))
    json_result = {
        "action": "finishedProcessing",
        "result": result
    }
    
    json_encoded_result = json.dumps(json_result)
    client.send(json_encoded_result.encode('utf-8'))
    client.close()
