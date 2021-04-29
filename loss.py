import torch

# The centroid loss
# outputs: Outputs from network. Tensor [[]]
# labels: Many-hot encoding of the labels. Tensor [[]]
# centroids: Vectors representing the centroids of each class. Tensor [[]]
# radii: Radius associated with the centroids of each class. Tensor []
# Returns: batch-averaged loss

# Try combining with cross entropy somehow??
# screw it, we going full bce

def centroid_loss(outputs,labels,centroids,radii,device=None,return_inferred=False):
    # For each embedding in batch, compute distance to centroids
    dist_to_centroid = torch.cdist(outputs, centroids, p=2)
    # print("dist to centroid")
    # print(dist_to_centroid)
    
    # Subtract radii so we can infer label
    # neg num = within that circle, pos num = outside that circle
    within_radii = dist_to_centroid - radii
    inferred_label = (within_radii < 0).float()

    # print("within radii")
    # print(within_radii)

    # print("labels")
    # print(labels)

    # Get zero indices from the labels
    # Turns zeros into -1
    # label_neg = torch.ones(labels.size())
    # label_neg[labels == 0] = -1
    
    # if device:
    #     within_radii = within_radii.cuda(device)
    #     label_neg = label_neg.cuda(device)
    

    # # print("label negated")
    # # print(label_neg)

    # # Compute loss contributions for each data point
    # loss_contributions = within_radii*label_neg
    # print("loss contribution")
    # print(loss_contributions)
    
    # Find the labels that are different in pred label and true label
    label_diff = inferred_label == labels
    label_diff = (~label_diff).float()
    # print(label_diff)

    # # Element-wise multiplication to get the final loss contribution from each centroid
    loss_contributions = within_radii*label_diff
    # # Flips negative distances to positive distances 
    loss_contributions = loss_contributions * (-1*(loss_contributions<0) + (loss_contributions>0))
    
    # In the future one could weight the loss contribution from each centroid (classes) differently
    # but for simplicity we assume each centroid contributes to the loss equally
    # loss_contributions = class_weights * loss_contributions

    # Experimental: Square every term then half them for easy derivative
    #loss_contributions = (1/2)*torch.square(loss_contributions)
    
    # Compute the loss for each data point
    loss_per_data_point = torch.sum(loss_contributions,axis=1)
    # print("loss per data point")
    # print(loss_per_data_point)
    
    # For simplicity sake, the final loss per batch will always be averaged by the batch size
    final_loss = torch.mean(loss_per_data_point)
    
    if return_inferred:
        return final_loss,inferred_label
        
    print(final_loss)

    return final_loss