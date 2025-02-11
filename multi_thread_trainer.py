
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
        if must_save_results:
            with open('multi_thread_results.json', 'w') as json_file:
                json.dump(results, json_file, indent=4)
        return results
    def process_ai_parameters(self,ai_params:AIParameters):
        print(f"Processing {ai_params.__str__()}")
        result = ai_params.train_ai("MYSELF", self.cnn)
        print(f"Finished processing {ai_params.__str__()}")
        self.fila_processamento.remove(ai_params)
        return result