import time
import torch
import torch.nn as nn
import copy
from loss import centroid_loss
from loss_util import infer_data_label
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# # Hyperparameter. Higher = more focus on centroid loss
# LAMBDA = 0.1

def train_model(model,dataloaders, optimizer, num_epochs=15, loss_func="centroid", centroids=None,radii=None,device=None,wandb=None,save_path=None):
    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    model.to(device)
    centroids = centroids.to(device)
    radii = radii.to(device)

    criterion = None
    if loss_func == "bce" or loss_func == "centroid":
        criterion = nn.BCEWithLogitsLoss()
        
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    
                    if loss_func=="centroid":
                        #preds = infer_data_label(outputs,centroids,radii)
                        #preds.requires_grad = True
                        
                        loss,preds = centroid_loss(outputs,labels,centroids,radii,device,return_inferred=True)
                        
#                         preds.requires_grad = True
#                         # print(outputs)
#                         # print(preds)
#                         # print(labels)
#                         bce_loss = criterion(preds, labels)
#                         loss = bce_loss + LAMBDA*loss
    
                        
                        #loss,preds = centroid_loss(outputs,labels,centroids,radii,return_inferred=True)
                    elif loss_func=="bce":
                        loss = criterion(outputs, labels)
                        # Use sigmoid instead of softmax so we can get more than 1 pred with > 0.5 prob
                        preds = torch.sigmoid(outputs)
                                        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                
                # TODO: Replace with better evaluation metric such as f1
                # Currently only strict matches are considered (labels must be exactly equal)
                #running_f1 = f1_score(outGT, outPRED > 0.5, average="samples")
                
                if loss_func == "bce":
                    preds_hard_cutoff = (preds>0.5).float() # we hard predict everything above 0.5 to be true
                    exact_matches = torch.sum(torch.eq(preds_hard_cutoff,labels.data),axis=1) == labels.data.shape[1]
                elif loss_func == "centroid":
                    exact_matches = torch.sum(torch.eq(preds,labels.data),axis=1) == labels.data.shape[1]
                
                running_corrects += torch.sum(exact_matches)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if wandb != None:
                if phase == 'train':
                    wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc})
                elif phase =='val':
                    wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc})
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    if save_path != None:
        torch.save(model.state_dict(), save_path)

    return model, val_acc_history

def eval_model(model,dataloader,loss_func="centroid",centroids=None,radii=None,device=None,report_all=False):
    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    model.to(device)
    centroids = centroids.to(device)
    radii = radii.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    if report_all:
        assert device != None
        all_preds = []
        all_labels = []

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            if loss_func=="centroid":
                loss,preds = centroid_loss(outputs,labels,centroids,radii,device,return_inferred=True)
            # elif loss_func=="bce":
            #     loss = criterion(outputs, labels)
            #     # Use sigmoid instead of softmax so we can get more than 1 pred with > 0.5 prob
            #     preds = torch.sigmoid(outputs)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        
        # TODO: Replace with better evaluation metric such as f1
        # Currently only strict matches are considered (labels must be exactly equal)
        #running_f1 = f1_score(outGT, outPRED > 0.5, average="samples")

        if loss_func == "centroid":
            exact_matches = torch.sum(torch.eq(preds,labels.data),axis=1) == labels.data.shape[1]
            if report_all:
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())

        running_corrects += torch.sum(exact_matches)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    if report_all:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds).reshape((len(dataloader.dataset),all_preds.shape[2]))
        all_labels = np.array(all_labels).reshape((len(dataloader.dataset),all_labels.shape[2]))
        all_f1 = f1_score(all_labels, all_preds, average="samples")
        all_prc = precision_score(all_labels, all_preds, average="samples")
        all_rec = recall_score(all_labels, all_preds, average="samples")
        return epoch_loss,epoch_acc,all_f1,all_prc,all_rec



    return epoch_loss,epoch_acc

