
from ai_parameters import AIParameters
from cnn import CNN


fila_processamento: list[AIParameters] = []

def add_ai_parameters(list: list[AIParameters]):
    fila_processamento.extend(list)

def process_single_instance(is_multiprocessing: bool, cnn: CNN):
    for ai_params in fila_processamento:
        result = ai_params.train_ai("DESKTOP", cnn)
        print(result)