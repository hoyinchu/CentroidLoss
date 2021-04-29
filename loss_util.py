import torch

# This is identical to the first part of the centroid loss

# Given the output of a network and corresponding centroid and radii
# Returns the many-hot encoding label that the output belongs to
# outputs: Outputs from network. Tensor [[]]
# centroids: Vectors representing the centroids of each class. Tensor [[]]
# radii: Radius associated with the centroids of each class. Tensor []
# Returns: the inferred many-hot encoded label

def infer_data_label(outputs,centroids,radii):
    # For each embedding in batch, compute distance to centroids
    dist_to_centroid = torch.cdist(outputs, centroids, p=2) 
    
    # Subtract radii so we can infer label
    # neg num = within that circle, pos num = outside that circle
    within_radii = dist_to_centroid - radii
    inferred_label = (within_radii < 0).float()
    return inferred_label