import time

from cnn import CNN

class AIParameters:
    
    def __init__(self, json_data):
        self.replicacoes = json_data.get('replicacoes', 10)
        self.model_name = json_data.get('model_name', "Alexnet")
        self.epoch = json_data.get('epoch', 0)
        self.learning_rate = json_data.get('learning_rate', 0)
        self.weight_decays = json_data.get('weight_decays', 0)
    
    def train_ai(self, computer_id:str, cnn:CNN):
        inicio = time.time()
        acc_media, rep_max = cnn.create_and_train_cnn(self.model_name,self.epoch,self.learning_rate,self.weight_decays,self.replicacoes)
        fim = time.time()
        duracao = fim - inicio
        return f"Treinamento finalizado por: {computer_id} {self.model_name}-{self.epoch}-{self.learning_rate}-{self.weight_decays}-Acurácia média: {acc_media} - Melhor replicação: {rep_max} - Tempo:{duracao}"
    
    def __str__(self):
        return f"AIParameters: {self.model_name}-{self.epoch}-{self.learning_rate}-{self.weight_decays}"