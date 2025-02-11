
import json
from client import connect_to_socket_server
from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import time

from coordinator.coordinator import get_ai_parameters_list
from single_instance_trainer import add_ai_parameters, process_single_instance


is_multiprocessing = False
is_distributed = False

def define_transforms(height, width):
    data_transforms = {
        'train' : v2.Compose([
                    v2.Resize((height,width)),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test'  : v2.Compose([
                    v2.Resize((height,width)),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    return data_transforms


def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/resumido/train/',transform=data_transforms['train'])
    validation_data = datasets.ImageFolder('./data/resumido/validation/',transform=data_transforms['test'])
    test_data = datasets.ImageFolder('./data/resumido/test/',transform=data_transforms['test'])
    return train_data, validation_data, test_data

if __name__ == '__main__':
    data_transforms = define_transforms(224,224)
    train_data, validation_data, test_data = read_images(data_transforms)
    cnn = CNN(train_data, validation_data, test_data,8)
    if(is_distributed):
        connect_to_socket_server()
        pass
    else:
        results = {}
        param_list = get_ai_parameters_list()
        add_ai_parameters(param_list)
        single_instance_results = process_single_instance(is_multiprocessing, cnn)
        results["single_instance"] = single_instance_results
        with open('results.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)
        pass
    
# #Esta é a parte do código que deve ser atualizada e distribuída
#     replicacoes = 10
#     model_names=['Alexnet']
#     epochs = [10]
#     learning_rates = [0.001]
#     weight_decays = [0]
#     inicio = time.time()
#     acc_media, rep_max = cnn.create_and_train_cnn(model_names[0],epochs[0],learning_rates[0],weight_decays[0],replicacoes)
#     fim = time.time()
#     duracao = fim - inicio
#     print(f"{model_names[0]}-{epochs[0]}-{learning_rates[0]}-{weight_decays[0]}-Acurácia média: {acc_media} - Melhor replicação: {rep_max} - Tempo:{duracao}")