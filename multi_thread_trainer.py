
import json
import multiprocessing
from ai_parameters import AIParameters
from cnn import CNN


class MultiThreadTrainer:
    fila_processamento: list[AIParameters] = []
    cnn: CNN = None
    
    def __init__(self, parameter_list: list[AIParameters]):
        self.fila_processamento = parameter_list
        pass

    def process(self, cnn: CNN, must_save_results: bool = False):
        results = []
        self.cnn = cnn
        print("Starting multi-thread training")
        # print each of the parameters combination
        for ai_params in self.fila_processamento:
            print(ai_params.__str__())
        with multiprocessing.Pool() as pool:
            results = pool.map(self.process_ai_parameters, self.fila_processamento)
        print("Finished multi-thread training")
        return results
    

    def process_ai_parameters(self,ai_params:AIParameters):
        print(f"Processing {ai_params.__str__()}")
        result = ai_params.train_ai("MYSELF", self.cnn)
        with open('multi_thread_results.json', 'w') as json_file:
                try:
                    existing_results = json.load(json_file)
                except json.JSONDecodeError:
                    existing_results = []
                existing_results.append(result)
                json_file.seek(0)
                json.dump(existing_results, json_file, indent=4)
                json_file.truncate()
        print(result)
        self.fila_processamento.remove(ai_params)
        return result