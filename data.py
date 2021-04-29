import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

from data_util import transform_cifar10_target

# A slightly modified version of the original CIFAR10. For more read documentation on transform_cifar10_target
class ModifiedCIFAR10(Dataset):
    def __init__(self, cifar10_original, transform=None, target_transform=None):
        self.cifar10_original = cifar10_original
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.cifar10_original.__len__()

    def __getitem__(self, idx):
        image,label = self.cifar10_original.__getitem__(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

# Returns a dictionary with two dataloader objects of the modified CIFAR10
# One for training (Key = "train") and one for testing (Key = "val")
def get_modified_cifar10_data_loader_dict(batch_size):
    # Pre-defined input size for resnet
    INPUT_SIZE = 224

    # Data augmentation and normalization for training
    # Just normalization for validation
    # Normalization values are hard coded as provided the original resnet paper
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Checking if the original CIFAR10 training set is downloaded...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True)
    print("Checking if the original CIFAR10 test set is downloaded...")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True)

    print("Applying transformation to both training and testing set")
    trainset_modified = ModifiedCIFAR10(trainset,data_transforms['train'],transform_cifar10_target)
    testset_modified = ModifiedCIFAR10(testset,data_transforms['val'],transform_cifar10_target)

    trainloader = torch.utils.data.DataLoader(trainset_modified, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset_modified, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    print("Dataloader dictionary complete")
    # Create training and validation dataloaders
    dataloaders_dict = {'train': trainloader, 'val': testloader}

    return dataloaders_dict