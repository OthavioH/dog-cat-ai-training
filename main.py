
import itertools
import json
from ai_parameters import AIParameters
from client import connect_to_socket_server
from cnn import CNN
import torch
from torchvision import datasets
from torchvision.transforms import v2
import time

from coordinator import *
from multi_thread_trainer import MultiThreadTrainer
from single_instance_trainer import add_ai_parameters, process_single_instance


is_multiprocessing = True
is_distributed = False
is_coordinator = False

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

def get_ai_parameters_list():
        model_names = ['Alexnet', 'VGG11', 'MobilenetV3Large', 'MobilenetV3Small', 'Resnet18', 'Resnet101', 'VGG19'] 
        epochs = [5, 10]
        learning_rates = [0.001, 0.0001, 0.00001]
        weight_decays = [0, 0.0001]

        # Generate a ;ost with all combinations of parameters
        param_combinations = list(itertools.product(model_names, epochs, learning_rates, weight_decays))
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

if __name__ == '__main__':
    if(is_distributed):
        if(is_coordinator):
            ai_coordinator = Coordinator()
            ai_coordinator.start_coordinator()
        else:
            connect_to_socket_server()
        pass
    elif is_multiprocessing:
        data_transforms = define_transforms(224,224)
        train_data, validation_data, test_data = read_images(data_transforms)
        cnn = CNN(train_data, validation_data, test_data,8)
        param_list = get_ai_parameters_list()
        add_ai_parameters(param_list)
        multi_thread_trainer = MultiThreadTrainer(param_list)
        results = multi_thread_trainer.process(cnn)
        with open('multi_thread_results.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)
    else:
        data_transforms = define_transforms(224,224)
        train_data, validation_data, test_data = read_images(data_transforms)
        cnn = CNN(train_data, validation_data, test_data,8)
        results = {}
        param_list = get_ai_parameters_list()
        add_ai_parameters(param_list)
        single_instance_results = process_single_instance(is_multiprocessing, cnn)
        results["single_instance"] = single_instance_results
        with open('single_instance_results.json', 'w') as json_file:
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