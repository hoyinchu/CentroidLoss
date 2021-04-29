import torch
import numpy as np

def transform_cifar10_target(label_num):
    # The original mappings are as follow:
    # airplane : 0
    # automobile : 1
    # bird : 2
    # cat : 3
    # deer : 4
    # dog : 5
    # frog : 6
    # horse : 7
    # ship : 8
    # truck : 9
    
    # The new mappings will have the following classes added:
    # Animals: 10, true when any of dogs, cats, horse, deer, bird, frog is true
    # Domesticated: 11, true when either dogs or cats are true
    # Vehicles: 12, true when any of automobiles, truck, plane, ship is true
    # Ground: 13, true when either automobiles or truck is true
    
    label_vec = [0.] * 14
    label_vec[label_num] = 1.
    
    animals_set = set([2,3,4,5,6,7])
    domesticated_set = set([3,5])
    vehicles_set = set([0,1,8,9])
    ground_set = set([1,9])
    
    animal_index = 10
    domesticated_index = 11
    vehicles_index = 12
    ground_index = 13
    
    if label_num in animals_set:
        label_vec[animal_index] = 1.
    if label_num in domesticated_set:
        label_vec[domesticated_index] = 1.
    if label_num in vehicles_set:
        label_vec[vehicles_index] = 1.
    if label_num in ground_set:
        label_vec[ground_index] = 1.
    
    y = torch.tensor(label_vec)
    
    return y

# Given the modified cifar10 dataset, sample n number of samples from a given class
def sample_from_modified_cifar10_dataset(dataset,total_samples,sample_class,random_state=0):
    sample_class_indices = np.where(np.array(dataset.cifar10_original.targets) == sample_class)[0]
    np.random.seed(random_state)
    sampling_indices = np.random.choice(sample_class_indices,total_samples)
    to_return = [dataset.__getitem__(i) for i in sampling_indices]
    return to_return

def make_2d_pred(model,data_to_pred,device):
    model.eval()
    all_outputs = []
    for data_point in data_to_pred:
        inputs,labels = data_point
        if device != None:
            inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs[None,...])
            if device != None:
                all_outputs.append(outputs.cpu().numpy()[0])
            else:                
                all_outputs.append(outputs[0])
    return np.array(all_outputs)
