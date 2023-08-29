
'''
collect all text files in given directory with metadata fields
cmd ine
find . -type f -exec file {} \; | grep ":.* ASCII text"

if we want to find all text files in the current directory, including its sub-directories, then we have to augment
the command using the Linux find command:
find . -type f -exec file -i {} \; | grep " text/plain;" | wc

'''

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_images(root_directory):
    target_size = (512,512)
    # Define transformations to apply to the images (resizing, normalization, etc.)
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize images to target size pixels
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet statistics
    ])

    # Create an ImageFolder dataset
    return ImageFolder(root=root_directory, transform=transform)

if __name__ == '__main__':

    root_directory = "./data/docs-sm"  # TODO args
    dataset = load_images(root_directory=root_directory)

    # Create a DataLoader for batching and shuffling
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Print some information about the dataset
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Number of images: {len(dataset)}")

    for batch in dataloader:
        images, labels = batch
        # Now you can use 'images' and 'labels' in your training loop

